# Security Policy

## Supported Versions

The following versions of MoonLab Quantum Simulator receive security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of MoonLab Quantum Simulator seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **security@tsotchke.ai**

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information in your report:

- Type of vulnerability (e.g., buffer overflow, injection, information disclosure)
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

After you submit a report, we will:

1. **Acknowledge receipt** within 48 hours
2. **Confirm the vulnerability** and determine its scope within 7 days
3. **Develop a fix** and prepare a security release
4. **Release the fix** and publicly disclose the vulnerability (with credit to you, unless you prefer to remain anonymous)

### Disclosure Policy

- We follow a **90-day disclosure timeline** from the initial report
- We will coordinate with you on the disclosure date
- If a fix takes longer than 90 days, we will work with you on an extended timeline
- We will credit you in the security advisory (unless you prefer anonymity)

## Security Best Practices

When using MoonLab Quantum Simulator in production:

### Memory Safety

- The simulator uses secure memory handling for sensitive quantum state data
- All allocations are zeroed on deallocation to prevent memory dumps
- Use the latest version to benefit from ongoing security improvements

### Entropy Sources

- The Quantum Random Number Generator (QRNG) uses hardware entropy sources
- NIST SP 800-90B compliant health tests are included
- Do not disable health tests in production environments

### API Security

When deploying the simulator as a service:

- Always use TLS/HTTPS for API communication
- Implement proper authentication and authorization
- Rate limit API endpoints to prevent abuse
- Validate all input parameters before simulation
- Set appropriate resource limits for simulations

### Container Security

When using the Docker image:

- Run containers as non-root (the default)
- Use read-only root filesystem where possible
- Keep container images updated
- Scan images for vulnerabilities before deployment

## Security Features

### Built-in Security

- **Secure Memory**: Quantum state data is securely zeroed on deallocation
- **Health Tests**: NIST-compliant tests validate entropy source quality
- **Input Validation**: All API inputs are validated for bounds and type safety
- **No Predictable Sources**: `rand()` and other predictable sources are prohibited

### Cryptographic Considerations

- The QRNG is suitable for generating cryptographic key material
- Bell test validation ensures true quantum randomness
- Hardware entropy pooling provides defense in depth

## Security Advisories

Security advisories will be published on:

- [GitHub Security Advisories](https://github.com/tsotchke/moonlab/security/advisories)
- The project's [security mailing list](mailto:security-announce@tsotchke.ai)

## Bug Bounty

We do not currently operate a bug bounty program. However, we deeply appreciate security researchers who take the time to report vulnerabilities responsibly.

## PGP Key

For sensitive communications, you may encrypt your message using our PGP key:

```
-----BEGIN PGP PUBLIC KEY BLOCK-----
[PGP key will be added here]
-----END PGP PUBLIC KEY BLOCK-----
```

## Contact

- Security issues: security@tsotchke.ai
- General questions: support@tsotchke.ai
- GitHub: https://github.com/tsotchke/moonlab

---

Thank you for helping keep MoonLab Quantum Simulator and our users safe!
