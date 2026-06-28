//! Thread-safe token-bucket rate limiter (since v1.0.3).
//!
//! Native Rust port of `src/utils/token_bucket.{c,h}`; semantics
//! match the C primitive's CHECK suite so the same admission-hook
//! policy ports between languages.
//!
//! Use case inside a Rust admission hook:
//!
//! ```no_run
//! use moonlab::admission_hook::{AdmissionDecision, AdmissionHook};
//! use moonlab::token_bucket::TokenBucket;
//! use std::collections::HashMap;
//! use std::sync::Mutex;
//!
//! let buckets: Mutex<HashMap<String, TokenBucket>> =
//!     Mutex::new(HashMap::new());
//!
//! let _hook = AdmissionHook::new(move |req| {
//!     let Some(tid) = req.tenant_id() else {
//!         return AdmissionDecision::Refused(-405);
//!     };
//!     let mut tab = buckets.lock().unwrap();
//!     let bkt = tab.entry(tid.to_string())
//!         .or_insert_with(|| TokenBucket::new(1000, 100));
//!     let cost = if req.num_shots() > 0 { req.num_shots() as u64 } else { 1 };
//!     if bkt.take(cost) {
//!         AdmissionDecision::Admitted
//!     } else {
//!         AdmissionDecision::Refused(-408)  // RATE_LIMITED
//!     }
//! });
//! ```

use std::sync::Mutex;
use std::time::Instant;

/// A simple, thread-safe token bucket.
///
/// Internal state is protected by a `Mutex` rather than the
/// C primitive's lock-free CAS loop -- python and rust admission
/// hooks fire on the worker thread of the C server, and contention
/// is bounded by `MAX_CONCURRENT`.  The mutex keeps the
/// implementation auditable and matches the python port's
/// semantics byte-for-byte.
pub struct TokenBucket {
    inner: Mutex<Inner>,
}

struct Inner {
    burst:   f64,
    refill:  f64,    // tokens/sec; 0 disables time-based refill
    tokens:  f64,    // current balance
    last:    Instant,
}

impl TokenBucket {
    /// Create a bucket that starts FULL at `burst` and refills at
    /// `refill_per_sec` tokens/second.  Pass `refill_per_sec == 0`
    /// to disable time-based refill (one-shot budget; replenish via
    /// `refill`).
    ///
    /// # Panics
    /// Panics if `burst == 0` -- a zero-capacity bucket is never
    /// useful and almost always a bug.
    pub fn new(burst: u64, refill_per_sec: u64) -> Self {
        assert!(burst > 0, "TokenBucket::new: burst must be > 0");
        TokenBucket {
            inner: Mutex::new(Inner {
                burst:  burst as f64,
                refill: refill_per_sec as f64,
                tokens: burst as f64,
                last:   Instant::now(),
            }),
        }
    }

    fn accrue(inner: &mut Inner, now: Instant) {
        if inner.refill == 0.0 { return; }
        let dt = now.saturating_duration_since(inner.last).as_secs_f64();
        if dt <= 0.0 { return; }
        inner.tokens = (inner.tokens + dt * inner.refill).min(inner.burst);
        inner.last = now;
    }

    /// Attempt to remove `n` tokens.  Returns `true` on success
    /// (tokens removed), `false` on insufficient balance (no
    /// tokens removed).  `n == 0` always succeeds.
    pub fn take(&self, n: u64) -> bool {
        if n == 0 { return true; }
        let mut inner = self.inner.lock().expect("token-bucket mutex poisoned");
        let now = Instant::now();
        Self::accrue(&mut inner, now);
        if inner.tokens < n as f64 {
            return false;
        }
        inner.tokens -= n as f64;
        true
    }

    /// Add `n` tokens, clipped at burst.
    pub fn refill(&self, n: u64) {
        if n == 0 { return; }
        let mut inner = self.inner.lock().expect("token-bucket mutex poisoned");
        inner.tokens = (inner.tokens + n as f64).min(inner.burst);
    }

    /// Read the current (lazily refilled) integer balance.
    pub fn peek(&self) -> u64 {
        let mut inner = self.inner.lock().expect("token-bucket mutex poisoned");
        let now = Instant::now();
        Self::accrue(&mut inner, now);
        inner.tokens as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn init_starts_full() {
        let b = TokenBucket::new(100, 10);
        assert_eq!(b.peek(), 100);
    }

    #[test]
    fn take_drains() {
        let b = TokenBucket::new(10, 0);
        for _ in 0..10 { assert!(b.take(1)); }
        assert!(!b.take(1));
        assert_eq!(b.peek(), 0);
    }

    #[test]
    fn refill_clips_at_burst() {
        let b = TokenBucket::new(5, 0);
        assert!(b.take(5));
        assert_eq!(b.peek(), 0);
        b.refill(1000);
        assert_eq!(b.peek(), 5);
    }

    #[test]
    fn time_refill_accrues() {
        let b = TokenBucket::new(100, 100);
        assert!(b.take(100));
        assert_eq!(b.peek(), 0);
        thread::sleep(Duration::from_millis(250));
        let p = b.peek();
        /* Verify invariants, not exact count.  CI runners can overrun
         * the sleep -- 250ms wall can become 400-500ms under load,
         * giving 40-50 tokens.  What matters is the floor (we got at
         * least the nominal accrual minus sleep underrun grace) and
         * the cap (we never exceed burst). */
        assert!(p >= 20, "after >=250ms at 100/s, peek = {} (expected >= 20)", p);
        assert!(p <= 100, "burst cap violated, peek = {} (expected <= 100)", p);
        assert!(b.take(10));
    }

    #[test]
    fn thread_safety_exact_balance() {
        let b = Arc::new(TokenBucket::new(500, 0));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let b = Arc::clone(&b);
            handles.push(thread::spawn(move || {
                let mut ok = 0;
                for _ in 0..1000 {
                    if b.take(1) { ok += 1; }
                }
                ok
            }));
        }
        let total: u32 = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(total, 500, "expected exactly 500 takes across 8 threads");
        assert_eq!(b.peek(), 0);
    }

    #[test]
    fn zero_cost_take_is_free() {
        let b = TokenBucket::new(1, 0);
        assert!(b.take(1));
        assert_eq!(b.peek(), 0);
        assert!(b.take(0));  // zero-cost: always succeeds
        assert_eq!(b.peek(), 0);
    }

    #[test]
    #[should_panic(expected = "burst must be > 0")]
    fn zero_burst_panics() {
        let _ = TokenBucket::new(0, 1);
    }
}
