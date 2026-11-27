from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pathlib import Path

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(2)
sec.left_margin = sec.right_margin = Cm(2.2)
normal = doc.styles['Normal']
normal.font.name = 'Arial'
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(7)
normal.paragraph_format.line_spacing = 1.12
for name, size in [('Title',23),('Subtitle',12),('Heading 1',16),('Heading 2',12)]:
    s=doc.styles[name]; s.font.name='Arial'; s.font.size=Pt(size); s.font.color.rgb=RGBColor(0,0,0)
    s.paragraph_format.space_before=Pt(12); s.paragraph_format.space_after=Pt(8)
footer=sec.footer.paragraphs[0]; footer.alignment=2
footer.add_run('Page ').font.size=Pt(9)
field=OxmlElement('w:fldSimple'); field.set(qn('w:instr'),'PAGE'); footer._p.append(field)
def p(t,style=None): return doc.add_paragraph(t,style)
def h(t): doc.add_heading(t,1)
def sub(t): doc.add_heading(t,2)
def page(): doc.add_page_break()
def table(headers, rows, widths):
    t=doc.add_table(rows=1, cols=len(headers)); t.autofit=False
    for c,w in zip(t.columns,widths): c.width=Cm(w)
    for i,row in enumerate([headers]+rows):
        cells=t.rows[0].cells if i==0 else t.add_row().cells
        for j,text in enumerate(row):
            cells[j].width=Cm(widths[j]); cells[j].text=text
            tcpr=cells[j]._tc.get_or_add_tcPr()
            borders=OxmlElement('w:tcBorders')
            for edge in ['top','left','bottom','right']:
                e=OxmlElement('w:'+edge); e.set(qn('w:val'),'single'); e.set(qn('w:sz'),'4'); e.set(qn('w:color'),'D9D9D9'); borders.append(e)
            tcpr.append(borders)
            margins=OxmlElement('w:tcMar')
            for side in ['top','left','bottom','right']:
                e=OxmlElement('w:'+side); e.set(qn('w:w'),'90'); e.set(qn('w:type'),'dxa'); margins.append(e)
            tcpr.append(margins)
            shade=OxmlElement('w:shd'); shade.set(qn('w:fill'),'E4ECF4' if i==0 else ('F6F6F6' if i%2==0 else 'FFFFFF')); tcpr.append(shade)
            for para in cells[j].paragraphs:
                para.paragraph_format.space_after=Pt(3)
                for r in para.runs: r.font.size=Pt(10); r.bold=(i==0)
        trpr=t.rows[i]._tr.get_or_add_trPr(); trpr.append(OxmlElement('w:cantSplit'))
        if i==0: trpr.append(OxmlElement('w:tblHeader'))
    p('')

p('RESEARCH PROGRESS REPORT','Title')
p('Image Forgery Localization Using U-Net with ResNet34','Subtitle')
p('Pre-thesis project  |  Updated 18 November 2026')
h('1 Overview and objectives')
p('This project develops a deep learning model to locate suspected manipulated regions in an image. The current model is U-Net with an ImageNet-pretrained ResNet34 encoder. Three input variants are investigated: RGB, RGB with noise residuals, and RGB with DCT frequency features.')
p('The project has reached model implementation and training. The next priorities are to verify all multi-seed results, complete test-set evaluation, and assess generalization before drawing conclusions about the effectiveness of each input variant.')
sub('Problem scope')
p('The approach is supervised deep learning for passive image forensics. It does not require a watermark embedded before manipulation. Its direct task is binary segmentation: assigning each pixel to a suspected manipulated region or background. Image-level detection, meaning a Real/Fake decision, requires an additional decision rule or prediction branch and separate evaluation.')
sub('Research questions')
for s in ['Do noise residuals or DCT features improve localization over RGB-only input under controlled experimental conditions?', 'Are results consistent across random seeds, and do additional features increase false alarms on authentic images?', 'Does performance transfer to datasets not used for training or model selection?']: p(s,'List Bullet')
sub('Target deliverables')
p('The pre-thesis report will include the method, a reproducible protocol, test results, evaluation on at least one dataset outside CASIA, failure analysis, and a single-image inference demo. The study does not assume in advance that Noise or DCT will outperform RGB.')

page(); h('2 Architecture and input variants')
p('ResNet34 acts as the encoder and extracts feature maps at multiple resolutions. The U-Net decoder progressively restores spatial resolution and combines encoder features through skip connections. A segmentation head converts the final features into one logit per pixel.')
table(['Component','Role'],[['ResNet34 encoder','Learns feature representations; initialized from ImageNet and trained for localization.'],['U-Net decoder','Restores spatial information and constructs features for pixel-level prediction.'],['Skip connections','Transfer encoder features to corresponding decoder stages.'],['Segmentation head','Produces a one-channel output; sigmoid converts logits into scores.']],[5,11.6])
sub('Three separately trained models')
table(['Variant','Channels','Features'],[['RGB','3','Three color channels normalized according to the training configuration.'],['RGB + Noise','6','RGB plus residuals calculated as RGB minus Gaussian-blurred RGB after resizing.'],['RGB + DCT','6','RGB plus three frequency-band energy maps from local DCT on 8 × 8 blocks.']],[4,2,10.6])
p('The current method combines features by concatenating input channels. It does not yet use two separate encoders or a dedicated fusion module. Residuals can contain edges and texture and should not be equated with PRNU sensor fingerprints. DCT is used for frequency analysis, not watermark embedding.')
sub('Position of the model within the project')
p('U-Net–ResNet34 is a CNN architecture, not SegFormer. ResNet34 is integrated as the U-Net encoder. Whether this satisfies the course requirement for model integration should be agreed with the supervisor. Replacing the complete model with SegFormer changes the architecture; it does not automatically integrate SegFormer with U-Net.')

page(); h('3 From an input image to a predicted mask')
table(['Step','Operation','Result'],[['1','Read and resize the image','An RGB image of 256 × 256 pixels.'],['2','Prepare features','Three or six channels using the same feature extraction and normalization as training.'],['3','ResNet34 encoder','Multi-resolution feature maps.'],['4','U-Net decoder and skip connections','Feature maps with restored spatial resolution.'],['5','Segmentation head','A one-channel logit map of 256 × 256 pixels.'],['6','Sigmoid','A score between 0 and 1 for every pixel.'],['7','Thresholding','A binary mask of suspected manipulation and background.']],[1.4,6,9.2])
sub('What happens after the decoder')
p('The decoder does not directly produce a black-and-white image. The segmentation head produces logits, which are raw scores that may be negative or positive. Sigmoid maps them to the range 0–1. For example, logits of -2.0, 0.4, and 2.2 correspond approximately to 0.12, 0.60, and 0.90. With a threshold of 0.5, these pixels receive labels 0, 1, and 1.')
p('A value of 1 denotes suspected manipulation; 0 denotes background. A sigmoid score should not be interpreted as calibrated confidence for the entire image. An empty predicted mask does not prove that an image is authentic.')
sub('Postprocessing for user outputs')
p('Map predictions back to the original image size using a fixed interpolation rule to produce an overlay. Extract connected components from the mask for bounding boxes. Compute the predicted area fraction as foreground pixels divided by all pixels. Use nearest-neighbor interpolation for binary masks; when resizing score maps, threshold after interpolation and document the protocol.')
sub('Direct and additional outputs')
p('The direct score output is a pixel-level map. Thresholding produces the mask. Boxes and area estimates are postprocessing outputs. An image-level Real/Fake decision requires a separately evaluated score aggregation rule or classification head. IoU and Dice can only be calculated when a ground-truth mask is available.')

page(); h('4 Training and the meaning of random seeds')
p('During training, predictions are compared with ground-truth masks. The loss is used to compute gradients, and the optimizer updates model weights. During inference, the model only receives an image and uses its learned weights without updating them.')
p('Training flow: image → model → logits → BCE plus Dice loss against the mask → backpropagation → weight update. Do not threshold predictions before calculating the loss. BCEWithLogits operates on logits; the Dice term uses an appropriate probability representation.')
table(['Parameter','Recorded protocol setting'],[['Current dataset','CASIA v2 and masks used by the project pipeline'],['Training and validation','Logs report 8,765 training images and 1,768 validation images'],['Input size and batch size','256 × 256; batch size 8'],['Learning rate and budget','0.0001; 10 epochs per run in the controlled protocol'],['Loss','Weighted BCE combined with Dice loss'],['Checkpoint selection','Highest validation micro Dice at threshold 0.5; lower validation loss breaks ties'],['Seeds','7, 21, and 42; three configurations correspond to nine training runs']],[5,11.6])
sub('What seeds 7 21 and 42 mean')
p('A seed initializes a random number generator. It affects operations such as initialization of non-pretrained weights, sample ordering, and augmentation. The values 7, 21, and 42 have no special meaning and are not epoch or image counts.')
p('Multiple seeds help assess result stability. Keep the same training, validation, and test splits across seeds in this study. Report individual runs and mean ± sample standard deviation rather than selecting only the best seed. Reproducibility also depends on libraries, hardware, and deterministic settings.')
p('Assess the 10-epoch budget using learning curves. If the budget is extended, apply the change consistently across configurations and document the protocol before test-set evaluation.')

page(); h('5 Current progress and remaining gaps')
table(['Item','Recorded status'],[['U-Net baseline','Implemented, with diagnostic experiments.'],['U-Net–ResNet34 with RGB Noise and DCT','Three variants implemented, with training sections.'],['Checkpointing and resuming','Saving and resuming training are implemented in the notebook.'],['Seeds 7 and 21','PDF logs report completion of six runs across three configurations.'],['Seed 42 and nine-run summary','Additional code is present; complete outputs remain unverified.'],['CASIA test','The PDF ends partway through a cell; no complete test table is available for confirmation.'],['Cross-dataset evaluation','No confirmed results yet.'],['Complete inference demo','A consistent end-to-end inference pipeline needs verification and completion.']],[6,10.6])
sub('Conclusions that cannot yet be drawn')
p('There is not enough evidence to identify RGB, Noise, or DCT as the best variant. Running inference on a new image does not establish reliability on arbitrary images. Drive checkpoints should be checked alongside configuration and training history before runs are accepted as complete.')
sub('Risks affecting the validity of results')
for s in ['Data leakage: source images, donors, manipulated derivatives, and duplicates may be related across splits. Hash checks alone cannot establish all source relationships.', 'Mismatched masks, incorrect foreground polarity, or unreliable annotations can invalidate both training and evaluation.', 'Resizing to 256 × 256 can remove details in small manipulated regions. Configurations must use comparable preprocessing and evaluation protocols.', 'Selecting checkpoints or thresholds using the test set compromises independent evaluation.']: p(s,'List Bullet')

page(); h('6 Evaluation and completion goals')
sub('Metrics to report')
p('Treat manipulated pixels as the positive class. TP denotes correctly detected manipulated pixels, FP denotes authentic pixels flagged as manipulated, and FN denotes missed manipulated pixels. IoU = TP / (TP + FP + FN). Dice = 2TP / (2TP + FP + FN). Dice and Pixel-F1 are equivalent when calculated using the same definitions and aggregation.')
p('Report micro IoU, micro Dice, mean Dice on tampered images, and the false-positive pixel rate on authentic images. Distinguish pooled pixel metrics from per-image averages and document empty-mask handling. For image-level Real/Fake evaluation, add a confusion matrix, precision, recall, and F1 on a set containing both classes.')
sub('Prioritized roadmap')
table(['Priority','Task','Completion criterion'],[['1','Audit existing runs','A status table covering configurations, histories, and checkpoints for all nine runs.'],['2','Freeze data and protocol','Verified masks, splits, leakage checks, and checkpoint/threshold selection rules.'],['3','Complete training','Missing runs completed, with learning curves and a validation table.'],['4','Evaluate CASIA test','Per-seed results, mean ± SD, and per-image outputs.'],['5','Evaluate cross-dataset transfer','At least one external dataset, fixed checkpoints and thresholds, no fine-tuning.'],['6','Complete demo and report','Masks, overlays, boxes, area estimates, image scores, and failure analysis.']],[1.9,4.9,9.8])
p('An indicative three-week schedule, subject to GPU availability, is: week one for run auditing and training completion; week two for test and cross-dataset evaluation; week three for the demo, failure analysis, report, and slides. This is a proposed schedule, not a measured runtime estimate.')
sub('Pre-thesis completion goal')
p('Complete a reproducible study of how Noise and DCT affect U-Net–ResNet34. Deliver verified result tables, 12–20 successful and failed examples selected using explicit criteria, an explanation of limitations, and a single-image inference pipeline. No course grade or accuracy level is guaranteed in advance.')

page(); h('7 Learning priorities and references')
sub('Suggested learning sequence')
table(['Order','Topic','Practical understanding check'],[['1','Detection and localization','Explain an image, its ground-truth mask, and its predicted mask.'],['2','Data and splits','Explain why source images and their manipulated derivatives need relationship-aware splitting.'],['3','U-Net and ResNet34','Draw the encoder, decoder, skip connections, and input/output dimensions.'],['4','Logits sigmoid and loss','Explain how training differs from inference.'],['5','Noise and DCT','Visualize all feature maps for the same image.'],['6','Metrics and seeds','Calculate IoU/Dice on small masks and interpret mean ± SD.'],['7','Generalization','Explain why good CASIA test results do not establish reliability on all real-world images.']],[1.7,5.1,9.8])
sub('Project source')
p('[1] Notebook v4 - Colab3.pdf, supplied for progress review. Relevant locations: pages 292–293 describe the protocol; page 304 records seed 7 and 21 logs; pages 306–311 contain seed 42 and summary code; pages 312–314 begin CASIA test evaluation.')
sub('Further reading')
p('[2] Segmentation Models PyTorch. Quick Start. Example of U-Net with a ResNet34 encoder. https://segmentation-models-pytorch.readthedocs.io/en/latest/quickstart.html')
p('[3] Xie et al. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers, 2021. Included for architectural comparison; it is not the current project model. https://arxiv.org/abs/2105.15203')
sub('Conclusion')
p('The project has implemented U-Net–ResNet34 with three input variants. The next priority is to turn existing training runs into independent, stable, and reproducible evaluation evidence. Transformer models or new fusion architectures should be considered after the current result set is complete, or when explicitly required by the supervisor.')

out=Path('outputs/report_word/UNet_ResNet34_Progress_Report_EN.docx')
doc.core_properties.title='Image Forgery Localization Using U-Net with ResNet34'
doc.core_properties.subject='Model architecture, workflow, progress, and pre-thesis goals'
for style in doc.styles:
    if style.type == 1:
        lang=OxmlElement('w:lang'); lang.set(qn('w:val'),'en-US'); style.element.get_or_add_rPr().append(lang)
doc.save(out)
print(out.resolve())

