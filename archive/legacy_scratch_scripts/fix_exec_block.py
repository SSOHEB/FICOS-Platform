with open('scratch/build_exp9_full_spec.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace exec_report block to use simple print statements instead of nested f-strings
old_exec_block = """exec_report = f'''
============================================================
EXPERIMENT 9 — PRODUCTION-SUPPORTED ECONOMIC BACKTEST
============================================================

Production registry:
[{promoted_pairs_str}]

2025 blind holdout:
Total cases                = {n_tot:,}
Production-supported cases = {n_sup:,} ({n_sup/n_tot*100:.1f}%)
Unsupported cases          = {n_uns:,} ({n_uns/n_tot*100:.1f}%)

Production-supported decision split:
NOW      = {now_p:.1f}%
WAIT     = {wait_p:.1f}%
FLEXIBLE = {flex_p:.1f}%

Always Spot:
Mean cost  = ${spot_m:,.2f}
Total cost = ${spot_tot/1e6:.3f}M

Naive Horizon-Wait:
Mean cost  = ${wait_m:,.2f}
Total cost = ${wait_tot/1e6:.3f}M

FICOS:
Mean cost  = ${ficos_m:,.2f}
Total cost = ${ficos_tot/1e6:.3f}M

FICOS vs Always Spot:
Aggregate cost difference = ${agg_diff:+,.0f}
Aggregate saving %        = {agg_pct:+.3f}%
95% CI                    = [{ci_lo:+.3f}%, {ci_hi:+.3f}%]

% decisions cheaper than spot = {cheaper_pct:.1f}%

Mean regret  = ${mean_reg:,.2f}
P90 regret   = ${p90_reg:,.2f}
Worst regret = ${worst_reg:,.2f}
95% CI       = [${ci_reg_supp[0]:,.2f}, ${ci_reg_supp[1]:,.2f}]

Primary economic conclusion:
{verdict}

Primary limitation:
{limitation}

============================================================
UNSUPPORTED / FALLBACK ANALYSIS
============================================================

Cases    = {n_uns:,} ({n_uns/n_tot*100:.1f}% of total 2025)
FLEXIBLE = {unsupp_flex_pct:.1f}%

Main fallback reasons:
- UNPROMOTED_PAIR: Horizons 7D, 14D, 30D for Cape, Panamax, Supramax, Handy are excluded/unregistered in registry/manifest.json due to walk-forward horizon signal decay.
- Production policy intentionally defaults to Index-Linked Floating Rate (FLEXIBLE) for unsupported pairs to prevent unvalidated directional bets.

============================================================
WHAT CHANGED FROM THE PREVIOUS EXPERIMENT 9
============================================================

1. EVALUATION POPULATION FIX:
   - Previous run evaluated only 7D, 14D, 30D (N=2,856). Since all 7D/14D/30D are unpromoted in registry/manifest.json, 100% of cases defaulted to FLEXIBLE (NOW=0%, WAIT=0%).
   - Corrected run evaluates the ACTUAL production-promoted pairs (1D for Cape, Panamax, Supramax, Handy; N=952) producing active NOW (10.2%), WAIT (9.6%), and FLEXIBLE (80.3%) decisions.

2. SEPARATION OF PRIMARY VS SECONDARY RESULTS:
   - Production-supported cases and unsupported fallback cases are now reported separately. Unsupported cases are no longer conflated with the production policy performance.

3. COST MODEL UNIT MAGNITUDE PRESERVED:
   - Retained the forensic audit fix: Daily TCE ($/day) x 20-day voyage duration (~$300k-$700k per voyage). Documented source as configs/cost_model.yaml.

============================================================
'''"""

new_exec_block = """report_lines = [
    "============================================================",
    "EXPERIMENT 9 — PRODUCTION-SUPPORTED ECONOMIC BACKTEST",
    "============================================================",
    "",
    f"Production registry: [{promoted_pairs_str}]",
    "",
    "2025 blind holdout:",
    f"Total cases                = {n_tot:,}",
    f"Production-supported cases = {n_sup:,} ({n_sup/n_tot*100:.1f}%)",
    f"Unsupported cases          = {n_uns:,} ({n_uns/n_tot*100:.1f}%)",
    "",
    "Production-supported decision split:",
    f"NOW      = {now_p:.1f}%",
    f"WAIT     = {wait_p:.1f}%",
    f"FLEXIBLE = {flex_p:.1f}%",
    "",
    "Always Spot:",
    f"Mean cost  = ${spot_m:,.2f}",
    f"Total cost = ${spot_tot/1e6:.3f}M",
    "",
    "Naive Horizon-Wait:",
    f"Mean cost  = ${wait_m:,.2f}",
    f"Total cost = ${wait_tot/1e6:.3f}M",
    "",
    "FICOS:",
    f"Mean cost  = ${ficos_m:,.2f}",
    f"Total cost = ${ficos_tot/1e6:.3f}M",
    "",
    "FICOS vs Always Spot:",
    f"Aggregate cost difference = ${agg_diff:+,.0f}",
    f"Aggregate saving %        = {agg_pct:+.3f}%",
    f"95% CI                    = [{ci_lo:+.3f}%, {ci_hi:+.3f}%]",
    "",
    f"% decisions cheaper than spot = {cheaper_pct:.1f}%",
    "",
    f"Mean regret  = ${mean_reg:,.2f}",
    f"P90 regret   = ${p90_reg:,.2f}",
    f"Worst regret = ${worst_reg:,.2f}",
    f"95% CI       = [${ci_reg_supp[0]:,.2f}, ${ci_reg_supp[1]:,.2f}]",
    "",
    f"Primary economic conclusion: {verdict}",
    "",
    f"Primary limitation: {limitation}",
    "",
    "============================================================",
    "UNSUPPORTED / FALLBACK ANALYSIS",
    "============================================================",
    "",
    f"Cases    = {n_uns:,} ({n_uns/n_tot*100:.1f}% of total 2025)",
    f"FLEXIBLE = {unsupp_flex_pct:.1f}%",
    "",
    "Main fallback reasons:",
    "- UNPROMOTED_PAIR: Horizons 7D, 14D, 30D for Cape, Panamax, Supramax, Handy are excluded/unregistered in registry/manifest.json due to walk-forward horizon signal decay.",
    "- Production policy intentionally defaults to Index-Linked Floating Rate (FLEXIBLE) for unsupported pairs to prevent unvalidated directional bets.",
    "",
    "============================================================",
    "WHAT CHANGED FROM THE PREVIOUS EXPERIMENT 9",
    "============================================================",
    "",
    "1. EVALUATION POPULATION FIX:",
    "   - Previous run evaluated only 7D, 14D, 30D (N=2,856). Since all 7D/14D/30D are unpromoted in registry/manifest.json, 100% of cases defaulted to FLEXIBLE (NOW=0%, WAIT=0%).",
    "   - Corrected run evaluates the ACTUAL production-promoted pairs (1D for Cape, Panamax, Supramax, Handy; N=952) producing active NOW (10.2%), WAIT (9.6%), and FLEXIBLE (80.3%) decisions.",
    "",
    "2. SEPARATION OF PRIMARY VS SECONDARY RESULTS:",
    "   - Production-supported cases and unsupported fallback cases are now reported separately. Unsupported cases are no longer conflated with the production policy performance.",
    "",
    "3. COST MODEL UNIT MAGNITUDE PRESERVED:",
    "   - Retained the forensic audit fix: Daily TCE ($/day) x 20-day voyage duration (~$300k-$700k per voyage). Documented source as configs/cost_model.yaml.",
    "",
    "============================================================",
]
exec_report = "\\n".join(report_lines)"""

text = text.replace(old_exec_block, new_exec_block)

with open('scratch/build_exp9_full_spec.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated build_exp9_full_spec.py with report_lines.")
