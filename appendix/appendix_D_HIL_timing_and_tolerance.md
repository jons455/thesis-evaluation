# Appendix D — HIL Timing and Hardware Logs

Raw clock counts (when available from `model.statistics.inference_clk`) and per-metric tolerance band tables.

## Phase 5 report (SC-6a, SC-6c)

```
PVP Phase 5 — HIL Deployment Feasibility
Host: 10.42.0.1:5001
Generated: 2026-02-26 17:28:14

--- SC-6a: Hardware Repeatability (R12 vs R13) ---
  Scenario: step_mid_speed_1500rpm_2A
    mae_i_q: R12=0.682788, R13=0.682788, dev=0.000000 -> PASS
    mae_i_d: R12=1.052379, R13=1.052379, dev=0.000000 -> PASS
    itae_i_q: R12=0.000850, R13=0.000850, dev=0.000000 -> PASS
    itae_i_d: R12=0.001319, R13=0.001319, dev=0.000000 -> PASS
    settling_time_i_q: R12=inf, R13=inf, dev=0.000000 -> PASS
    overshoot: R12=60.664368, R13=60.664368, dev=0.000000 -> PASS
  Scenario: multi_step_bidirectional_1500rpm
    mae_i_q: R12=1.030733, R13=1.030733, dev=0.000000 -> PASS
    mae_i_d: R12=0.627326, R13=0.627326, dev=0.000000 -> PASS
    itae_i_q: R12=0.001695, R13=0.001695, dev=0.000000 -> PASS
    itae_i_d: R12=0.000196, R13=0.000196, dev=0.000000 -> PASS
    settling_time_i_q: R12=inf, R13=inf, dev=0.000000 -> PASS
    overshoot: R12=72.209656, R13=72.209656, dev=0.000000 -> PASS
  SC-6a: PASS

--- SC-6c: Timing Characterization ---
  step_mid_speed_1500rpm_2A: round-trip=3.682 ms (p95=6.222), chip=380.1 us
    NOTE: round-trip (3.682 ms) > control timestep (0.1 ms)
  multi_step_bidirectional_1500rpm: round-trip=3.652 ms (p95=6.242), chip=383.5 us
    NOTE: round-trip (3.652 ms) > control timestep (0.1 ms)
  SC-6c: Reported (non-gating)

NOTE: SC-6b (hardware-software agreement) requires R11 (Akida sim) data. Compare R12 metrics against R11 manually or in a separate script.

Overall Phase 5: SC-6a=PASS, SC-6c=Reported
```

**Evaluation (Phase 5 latency).** The infrastructure computes and logs P95/P99 and max round-trip latency per scenario. For the thesis run, the 95th-percentile round-trip latency is ~6.2 ms and the 99th-percentile reaches ~10–16 ms across scenarios; tail-latency spikes of up to ~300 ms were observed (e.g. 300.8 ms in R13 on `multi_step_bidirectional_1500rpm`). These values confirm that the communication overhead is not merely a mean-value artifact but exhibits significant variance attributable to the TCP/OS layer; the Discussion’s mention of ~300 ms tail spikes is grounded in this Phase 5 evaluation.

## Tolerance band check table (SC-6a: R12 vs R13)

| Scenario | Metric | R12 | R13 | Within band? |
| --- | --- | --- | --- | --- |
| step_mid_speed_1500rpm_2A | mae_i_q | 0.6828 | 0.6828 | (see report) |
| step_mid_speed_1500rpm_2A | itae_i_q | 0.000850 | 0.000850 | (see report) |
| step_mid_speed_1500rpm_2A | settling_time_i_q | — | — | (see report) |
| step_mid_speed_1500rpm_2A | overshoot | 60.66 | 60.66 | (see report) |
| multi_step_bidirectional_1500rpm | mae_i_q | 1.031 | 1.031 | (see report) |
| multi_step_bidirectional_1500rpm | itae_i_q | 0.001695 | 0.001695 | (see report) |
| multi_step_bidirectional_1500rpm | settling_time_i_q | — | — | (see report) |
| multi_step_bidirectional_1500rpm | overshoot | 72.21 | 72.21 | (see report) |

**SC-6b (R12 vs R11):** R11 is produced by Phase 5-Q (Akida software simulation). Compare R12 metrics to R11 manually or via a separate script; add table here when available.

**Network-attached Akida:** If the lab uses a remote board over TCP, describe host, port, and transfer metric in this section.