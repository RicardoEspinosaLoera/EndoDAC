# Response to the reviewers — second revision

**Manuscript:** Illumination-Invariant Foundation Model Adaptation for Self-Supervised Monocular Depth Estimation in Endoscopy

We thank the reviewers for their careful reading of the revised manuscript. All the remaining comments have been addressed. In the revised manuscript, the text added or reworded in this round is in red; the changes of the previous round are now in black. No result reported in the previous version has been modified: the changes consist of one new table, computed by inference only, and of clarifications and rewording.

---

## Reviewer #2

### Comment (1)

> The main concern is whether the proposed illumination-invariant components provide a sufficiently demonstrated contribution beyond the pretrained foundation model. […] More importantly, their contributions have not been directly isolated on the cross-domain datasets, where illumination robustness is expected to be particularly valuable.

**Response.** We agree that the previous version isolated the two illumination components on SCARED only. We have now isolated them directly on Hamlyn and C3VD. The new Table 9 (Section 4.1.2) reports, by inference only and with models trained on SCARED, the paired difference in AbsRel between two configurations that differ only in the component considered, with its 95% bootstrap interval and the number of test units in which the component helps.

The results are reported as they are:

- With the foundation encoder, the lighting calibration lowers the error on C3VD in all seven scenes (−0.0260), and has no measurable effect on the four Hamlyn sequences (+0.0002).
- With the foundation encoder, the illumination-invariant loss brings no measurable cross-domain benefit (+0.0001 on Hamlyn; −0.0034 on C3VD, not significant).
- With the ResNet-18 encoder, the differences on Hamlyn are slightly favourable (−0.0013 for the calibration, −0.0012 for the loss, −0.0017 for their combination) but not significant with four sequences, and the calibration alone and the combination increase the error on C3VD (+0.0149 and +0.0100).

We therefore do not claim a consistent cross-domain benefit of the two components. The manuscript now states that the only significant gain is that of the lighting calibration on C3VD with the foundation encoder, that the transfer benefit suggested for lightweight encoders by the indirect comparison of Section 5 is not confirmed by this direct analysis, and that most of the accuracy of MonoIIF comes from the pretrained encoder and its adaptation.

For the in-domain evaluation, a new summary table in Section 3.4 gives, from the values of Table 2, the change brought by the illumination components to each of the three proposed models with respect to the same backbone trained without them. The pattern is the same for the three backbones: SqRel decreases by 10% to 17% and RMSE by 5% to 9% (for MonoIIF, −15.6% and −8.7%), whereas the change in AbsRel (0.002 to 0.004) is of the order of the seed-to-seed variability. As each configuration corresponds to a single training run, the manuscript presents these reductions as a trend that is consistent across backbones, which suggests that the components mainly act on the largest depth errors, and not as an established effect.

**Changes in the manuscript.** New summary table and paragraph "Summary for the three proposed models" (Section 3.4); new Table 9 and new paragraph "Isolated contribution of the illumination components outside the training domain" (Section 4.1.2); Section 3.4 (the expectation of a cross-domain benefit is now said to be only partly confirmed); Section 5, paragraph "Cross-dataset generalization" (the statement that the components were isolated only on SCARED is replaced by the direct analysis); abstract and conclusion (one sentence each).

### Comment (2)

> The evaluation protocol also needs clarification. […] It is therefore unclear whether the test set influenced model selection. Furthermore, the statistical analysis, although improved, should be interpreted more cautiously. The absence of a statistically significant difference does not establish equivalence, especially with the limited number of test sequences.

**Response.**

*(a) Use of the SCARED test images.* The reviewer is right: the test images did influence model selection, and the revised manuscript now says so. We use the public SCARED split (15,351 training, 1,705 validation and 551 test images). Only the training images are used to optimize the network weights, and the validation images are used only to monitor a validation loss. The ablation study was evaluated on the 551 test images, and its results were used to choose the weight λ₁ and the three retained configurations. In addition, MonoIIF is trained with the publicly released training code of EndoDAC, which evaluates the model on the test split after each epoch and retains the epoch with the lowest RMSE; we kept this procedure unchanged, so that MonoIIF and EndoDAC are compared under the same training and selection procedure.

The SCARED values of the proposed models may therefore be slightly optimistic, and this is now stated in Sections 3.1 and 3.4. Two elements limit the consequences:

- The differences between the candidate configurations are of the order of the seed-to-seed variability (standard deviation of 0.0023 in AbsRel for MonoIIF), so no in-domain conclusion of the paper relies on the selected values. In particular, the manuscript does not claim that MonoIIF outperforms HADepth or EndoDAC on SCARED.
- The Hamlyn and C3VD datasets were used for inference only. No weight, hyperparameter or configuration was chosen with them, so the generalization results are independent of this selection.

*(b) Interpretation of the statistical analysis.* We agree that the absence of a significant difference does not establish equivalence. The terms "equivalent", "on par" and "statistically indistinguishable" have been removed from the abstract, Sections 1.3, 4.1 and 5 and the conclusion, and replaced by "no significant difference is detected". Section 3.2 now states that no equivalence test was performed and that, with four to seven test sequences, the available data only show that the methods cannot be separated. The comparisons are presented as exploratory.

**Changes in the manuscript.** Section 3.1 (public split and use of each subset); Section 3.2 (definition of "not significant", absence of an equivalence test); Section 3.4, first paragraph (model selection); abstract, Sections 1.3, 4.1, 5 and 6 (wording).

### Comment (3)

> The additional 3D surface reconstruction results are useful, but the conclusion that MonoIIF produces more geometrically accurate and coherent surfaces is still supported mainly by qualitative examples. The manuscript also needs a final consistency check, as some statements in the submission abstract and figure captions do not match the revised results or the authors' responses.

**Response.**

*(a) 3D surface reconstruction.* We agree. The aim of Section 4.3 is not to measure the accuracy of the reconstructed surfaces, but to check visually the continuity of their textures and the presence of geometric discontinuities, and the section now says so in its first paragraph. All the statements that drew a conclusion on accuracy from these examples, or that attributed the observed differences to the illumination-invariant loss, have been removed or reworded. The observations are now explicitly restricted to the examples shown ("in this example"), and the section concludes that these are visual illustrations on a few images and not a quantitative comparison, which do not establish that the surfaces obtained with MonoIIF are more accurate than those obtained with HADepth. The corresponding sentences of the abstract and of the discussion have been modified accordingly; the discussion now states that a quantitative evaluation of the reconstructed surfaces would be required to confirm these observations.

*(b) Consistency check.* The manuscript was checked against the results, and the following statements were corrected:

- Abstract: the unresolved wording on the comparison with HADepth and EndoDAC on SCARED was fixed, and the sentence on the 3D reconstructions was reworded as described above. The abstract entered in the submission system has been updated to match.
- Captions of the two error-map figures and corresponding text (Section 4.2): they no longer state that MonoIIF produces the lowest error; for C3VD they now recall that HADepth is more accurate on this dataset as a whole (Table 10).
- Sections 1.3 and 1.4: "competitive or superior accuracy" and "superior performance" were replaced by statements supported by the results.
- Section 3: "two proposed models" corrected to "three"; caption of Table 1 ("Five metrics") and the description of the thresholds δ<1.25² and δ<1.25³ corrected; a sentence that associated the weight λ₂ with the photometric loss corrected.
- Section 5: a cross-reference pointing to the wrong section was corrected.
- Conclusion: rewritten to cover the three variants and the results as they are now reported.

---

## Reviewer #3

### Comment (1)

> Figure 4 caption still reads "Architecture of the transformer-based depth estimation framework" rather than "Architecture of MonoIIT," as requested and acknowledged in the response letter.

**Response.** We apologize for this omission. The caption of Figure 4 now reads "Architecture of MonoIIT", and the title inside the figure was changed in the same way. The label "Conv-steam" was also corrected to "Conv-stem" in the figure and in its caption.

### Comment (2)

> MonoII architecture description: Although the MonoII backbone is now described in Section 3.5, the corresponding description is still absent from Section 2.2.4.

**Response.** A short paragraph "CNN backbone (MonoII)" has been added at the beginning of Section 2.2.4, before the descriptions of MonoIIT and MonoIIF. It describes the ResNet-18 encoder pretrained on ImageNet, the DispNet-like decoder and the ResNet-18 pose network, consistently with Section 3.5, to which it refers.

### Comment (3)

> Regarding the statistical analysis, I recommend avoiding terminology that implies statistical equivalence when the results only show a lack of statistically significant differences.

**Response.** We agree, and this terminology has been removed throughout the manuscript. "Statistically equivalent", "on par", "statistically indistinguishable" and "should be considered equivalent" were replaced by "no significant difference is detected" in the abstract, Sections 1.3, 4.1 and 5 and the conclusion. Section 3.2 now states explicitly that no equivalence test was performed and that the absence of a significant difference does not establish equivalence with four to seven test sequences (see also our response to comment (2) of Reviewer #2).

---

## Other change

In the overview figure of the framework (Figure 2), the label of the depth block was changed from "Transformer-based Depth Estimation" to "Foundation-based Depth Estimation", so that the figure corresponds to MonoIIF, the main model of the paper.
