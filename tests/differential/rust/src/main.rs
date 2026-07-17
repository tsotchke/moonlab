//! Cross-binding differential: the Rust `moonlab` crate's dense `QuantumState`
//! vs the numpy reference oracle pinned in the corpus.
//!
//! Reproduces every corpus circuit through the safe Rust binding and checks the
//! full probability vector and <Z_i>/<Z_iZ_j> expectations against the reference
//! within 1e-10 -- catching qubit-order / angle-sign / marshalling bugs across
//! the Rust FFI boundary that a C-only test cannot see.
//!
//! Reads the flat corpus.txt mirror (std-only parsing; no serde dependency).
//!
//! Usage:  diff_rust <corpus.txt>
//! Emits:  DIFF_RUST_RESULT status=PASS|FAIL cases=N failed=M
//! Exit:   0 pass, 1 fail, 2 usage/parse error.

use std::env;
use std::fs;
use std::process::exit;

use moonlab::QuantumState;

const TOL: f64 = 1e-10;

struct Gate {
    name: String,
    q0: i64,
    q1: i64,
    q2: i64,
    angle: f64,
}

struct Case {
    id: String,
    seed: String,
    num_qubits: usize,
    gates: Vec<Gate>,
    ref_prob: Vec<f64>,
    ref_z: Vec<f64>,
    zz: Vec<(usize, usize, f64)>,
}

struct Tokens<'a> {
    toks: Vec<&'a str>,
    i: usize,
}

impl<'a> Tokens<'a> {
    fn new(s: &'a str) -> Self {
        Tokens { toks: s.split_whitespace().collect(), i: 0 }
    }
    fn next(&mut self) -> Option<&'a str> {
        let t = self.toks.get(self.i).copied();
        if t.is_some() {
            self.i += 1;
        }
        t
    }
    fn expect(&mut self, want: &str) -> bool {
        matches!(self.next(), Some(t) if t == want)
    }
    fn u(&mut self) -> usize {
        self.next().unwrap().parse().unwrap()
    }
    fn i(&mut self) -> i64 {
        self.next().unwrap().parse().unwrap()
    }
    fn f(&mut self) -> f64 {
        self.next().unwrap().parse().unwrap()
    }
    fn s(&mut self) -> String {
        self.next().unwrap().to_string()
    }
}

fn parse(text: &str) -> Result<Vec<Case>, String> {
    let mut t = Tokens::new(text);
    if !t.expect("CORPUS") {
        return Err("bad header".into());
    }
    let _ver = t.u();
    let _seed = t.s();
    let _ncases = t.u();
    let mut cases = Vec::new();
    loop {
        match t.next() {
            Some("END") | None => break,
            Some("CASE") => {}
            Some(other) => return Err(format!("unexpected token {other}")),
        }
        let id = t.s();
        let _class = t.s();
        let n = t.u();
        let _depth = t.u();
        let _cliff = t.u();
        let seed = t.s();
        let ngates = t.u();
        let mut gates = Vec::with_capacity(ngates);
        for _ in 0..ngates {
            if !t.expect("G") {
                return Err("expected G".into());
            }
            gates.push(Gate {
                name: t.s(),
                q0: t.i(),
                q1: t.i(),
                q2: t.i(),
                angle: t.f(),
            });
        }
        if !t.expect("PROB") {
            return Err("expected PROB".into());
        }
        let np = t.u();
        let ref_prob: Vec<f64> = (0..np).map(|_| t.f()).collect();
        if !t.expect("EXPZ") {
            return Err("expected EXPZ".into());
        }
        let nz = t.u();
        let ref_z: Vec<f64> = (0..nz).map(|_| t.f()).collect();
        if !t.expect("EXPZZ") {
            return Err("expected EXPZZ".into());
        }
        let nzz = t.u();
        let zz: Vec<(usize, usize, f64)> = (0..nzz).map(|_| (t.u(), t.u(), t.f())).collect();
        if !t.expect("ENDCASE") {
            return Err("expected ENDCASE".into());
        }
        cases.push(Case { id, seed, num_qubits: n, gates, ref_prob, ref_z, zz });
    }
    Ok(cases)
}

fn apply(st: &mut QuantumState, g: &Gate) {
    let q0 = g.q0 as usize;
    let q1 = g.q1 as usize;
    let q2 = g.q2 as usize;
    match g.name.as_str() {
        "h" => { st.h(q0); }
        "x" => { st.x(q0); }
        "y" => { st.y(q0); }
        "z" => { st.z(q0); }
        "s" => { st.s(q0); }
        "sdg" => { st.sdg(q0); }
        "t" => { st.t(q0); }
        "tdg" => { st.tdg(q0); }
        "rx" => { st.rx(q0, g.angle); }
        "ry" => { st.ry(q0, g.angle); }
        "rz" => { st.rz(q0, g.angle); }
        "p" => { st.phase(q0, g.angle); }
        "cx" => { st.cnot(q0, q1); }
        "cz" => { st.cz(q0, q1); }
        "swap" => { st.swap(q0, q1); }
        "cp" => { st.cphase(q0, q1, g.angle); }
        "ccx" => { st.toffoli(q0, q1, q2); }
        other => panic!("unknown gate {other}"),
    }
}

fn exp_z(probs: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|qq| {
            probs
                .iter()
                .enumerate()
                .map(|(i, &p)| if (i >> qq) & 1 == 1 { -p } else { p })
                .sum()
        })
        .collect()
}

fn exp_zz(probs: &[f64], pairs: &[(usize, usize)]) -> Vec<f64> {
    pairs
        .iter()
        .map(|&(a, b)| {
            probs
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    let sa = if (i >> a) & 1 == 1 { -1.0 } else { 1.0 };
                    let sb = if (i >> b) & 1 == 1 { -1.0 } else { 1.0 };
                    sa * sb * p
                })
                .sum()
        })
        .collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <corpus.txt>", args[0]);
        exit(2);
    }
    let text = match fs::read_to_string(&args[1]) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {}: {e}", args[1]);
            exit(2);
        }
    };
    let cases = match parse(&text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("corpus parse error: {e}");
            exit(2);
        }
    };

    let mut exercised = 0usize;
    let mut failures: Vec<(String, String, &'static str, f64)> = Vec::new();

    for c in &cases {
        let mut st = match QuantumState::new(c.num_qubits) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  FAIL case={} QuantumState::new failed: {e:?}", c.id);
                failures.push((c.id.clone(), c.seed.clone(), "alloc", f64::INFINITY));
                continue;
            }
        };
        for g in &c.gates {
            apply(&mut st, g);
        }
        let probs = st.probabilities();
        exercised += 1;

        let dev = probs
            .iter()
            .zip(&c.ref_prob)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if dev > TOL {
            failures.push((c.id.clone(), c.seed.clone(), "prob", dev));
            continue;
        }
        let pairs: Vec<(usize, usize)> = c.zz.iter().map(|&(a, b, _)| (a, b)).collect();
        let ez = exp_z(&probs, c.num_qubits);
        let dz = ez
            .iter()
            .zip(&c.ref_z)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if dz > TOL {
            failures.push((c.id.clone(), c.seed.clone(), "expZ", dz));
        }
        let ezz = exp_zz(&probs, &pairs);
        let dzz = ezz
            .iter()
            .zip(&c.zz)
            .map(|(a, &(_, _, v))| (a - v).abs())
            .fold(0.0, f64::max);
        if dzz > TOL {
            failures.push((c.id.clone(), c.seed.clone(), "expZZ", dzz));
        }
    }

    for (cid, seed, what, dev) in failures.iter().take(50) {
        eprintln!(
            "  FAIL  case={cid} seed={seed} {what} [rust] vs reference dev={dev:.3e} (>{TOL:.0e})"
        );
    }
    let status = if failures.is_empty() { "PASS" } else { "FAIL" };
    println!(
        "DIFF_RUST_RESULT status={status} cases={exercised} failed={}",
        failures.len()
    );
    exit(if failures.is_empty() { 0 } else { 1 });
}
