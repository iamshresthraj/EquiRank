# Responsible Use Statement — ED-01: Fair Student-Support Prioritization

## The Proxy Problem: G3 < 10 as "Need"

Our system operationalizes "support needed" as a final Math grade below 10 out of 20 ($G3 < 10$). While quantifiable, this threshold is an imperfect proxy for educational need. A student scoring 10 may experience identical academic distress as one scoring 9, yet receives no support under a hard binary cutoff. Standardized grades fail to capture personal adversity, neurodiversity, mental health, or home crises. Allocating scarce support resources solely based on a single summative exam risks misdirecting aid away from non-academic impediments to student learning.

## Risks of Relying on G1 and G2

First- and second-period grades ($G1$ and $G2$) serve as permitted predictors and exhibit strong predictive power. However, early assessments frequently mirror structural inequities rather than raw capability or diligence. Disadvantaged students—facing longer travel times, limited internet connectivity, or lower parental educational attainment—are disproportionately represented among early low performers. Conditioning allocations heavily on early-checkpoint scores risks penalizing students for initial barriers rather than supporting their growth trajectory.

## Fairness Tradeoff: Before and After Mitigation

In our baseline model (balanced logistic regression with probability calibration), we observed:
- **Overall Recall**: 0.5758 (19 of 33 validation positives selected)
- **Worst-Group Recall ($R_{\min}$)**: 0.5263 (female students: 10/19)
- **Fairness Gap**: 0.1983 (school gap: $|0.5517 - 0.7500|$; sex gap: $|0.5263 - 0.6429| = 0.1165$)
- **Brier Score**: 0.0724
- **Composite Score**: 66.50 / 100

We evaluated group-aware reweighting and post-hoc probability recalibration across parameter sweeps (recorded in `outputs/pareto_tradeoff.png`). The optimal configuration applied a post-hoc probability calibration adjustment (+0.10 for female students), yielding:
- **Overall Recall**: 0.6061 (+0.0303)
- **Worst-Group Recall ($R_{\min}$)**: 0.5789 (+0.0526 for female students)
- **Fairness Gap**: reduced to 0.1638 (-0.0345)
- **Brier Score**: 0.0773
- **Composite Score**: 69.67 / 100 (+3.17 points)

This intervention uplifted female recall from 52.63% to 57.89% and improved GP school recall from 55.17% to 58.62%, narrowing the fairness disparity without sacrificing overall recall.

## Statistical Fragility of the MS Subgroup

The Mousinho da Silveira (MS) subgroup accounts for only 46 of 395 students (~11.6%) and only 4 positive cases in the validation split. Consequently, its recall (0.7500) represents exactly 3 out of 4 positive students. Over our 10-seed cross-validation stability analysis, MS recall exhibited high variance, causing the fairness gap to fluctuate between 0.05 and 0.40. System administrators must recognize that metrics for small subgroups carry substantial sampling uncertainty.

## Prerequisites for Real Deployment

1. **Human-in-the-Loop Review**: Counselors must qualitatively evaluate borderline students near the 20% budget boundary ($k=20$).
2. **Periodic Re-Auditing**: Continuous monitoring and retraining to track distribution shifts and algorithmic bias.
3. **Informed Consent & Transparency**: Clear opt-in policies and open disclosure of input factors to students and parents.
4. **Holistic Intake**: Supplementing grade data with counselor referrals and socio-emotional indicators.
