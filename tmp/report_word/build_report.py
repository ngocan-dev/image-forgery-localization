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
footer.add_run('Trang ').font.size=Pt(9)
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

p('BÁO CÁO TIẾN ĐỘ NGHIÊN CỨU','Title')
p('Định vị vùng giả mạo ảnh bằng U-Net kết hợp ResNet34','Subtitle')
p('Tiền luận văn  |  Cập nhật ngày 18 tháng 11 năm 2026')
h('1 Tổng quan và mục tiêu')
p('Dự án xây dựng mô hình học sâu để xác định vị trí nghi bị chỉnh sửa trong một ảnh. Mô hình chính hiện tại là U-Net với encoder ResNet34 pretrained trên ImageNet. Ba biến thể đầu vào được khảo sát gồm RGB, RGB kết hợp noise residual và RGB kết hợp đặc trưng tần số DCT.')
p('Tiến độ hiện tại đã bao gồm triển khai mô hình và các lượt huấn luyện. Trọng tâm tiếp theo là xác nhận đầy đủ kết quả nhiều seed, hoàn thiện đánh giá test và kiểm tra khả năng tổng quát hóa trước khi kết luận về hiệu quả của từng biến thể.')
sub('Phạm vi bài toán')
p('Hướng tiếp cận thuộc giám định ảnh thụ động bằng học sâu có giám sát. Mô hình không yêu cầu watermark được nhúng trước. Nhiệm vụ trực tiếp là binary segmentation: gán mỗi pixel vào vùng nghi giả hoặc vùng nền. Detection cấp ảnh, tức quyết định Real/Fake, cần một quy tắc hoặc nhánh dự đoán và được đánh giá riêng.')
sub('Câu hỏi nghiên cứu')
for s in ['Noise residual hoặc DCT có cải thiện localization so với RGB-only trong cùng điều kiện thực nghiệm không?', 'Kết quả có ổn định qua nhiều seed và có làm tăng báo giả trên ảnh thật không?', 'Hiệu quả có duy trì trên dataset chưa tham gia huấn luyện và lựa chọn mô hình không?']: p(s,'List Bullet')
sub('Sản phẩm mục tiêu')
p('Báo cáo tiền luận văn gồm mô tả phương pháp, protocol tái lập được, bảng kết quả test, đánh giá trên ít nhất một dataset ngoài CASIA, phân tích lỗi và demo dự đoán một ảnh. Không đặt trước kết luận rằng thêm Noise hoặc DCT chắc chắn tốt hơn RGB.')

page(); h('2 Kiến trúc và các biến thể đầu vào')
p('ResNet34 làm encoder trích xuất feature maps ở nhiều độ phân giải. Decoder U-Net tăng dần độ phân giải và kết hợp đặc trưng từ encoder qua skip connections. Segmentation head chuyển các đặc trưng cuối thành một logit cho mỗi pixel.')
table(['Thành phần','Vai trò'],[['ResNet34 encoder','Học biểu diễn đặc trưng; khởi tạo từ ImageNet rồi huấn luyện cho localization.'],['U-Net decoder','Khôi phục thông tin không gian và tạo biểu diễn phục vụ dự đoán pixel.'],['Skip connections','Đưa đặc trưng từ encoder sang decoder ở các mức tương ứng.'],['Segmentation head','Tạo đầu ra một kênh; sigmoid chuyển logits thành điểm số.']],[5,11.6])
sub('Ba mô hình được huấn luyện riêng')
table(['Biến thể','Kênh','Đặc trưng'],[['RGB','3','Ba kênh màu được chuẩn hóa theo cấu hình training.'],['RGB + Noise','6','RGB cùng residual tính bằng RGB trừ Gaussian blur sau resize.'],['RGB + DCT','6','RGB cùng ba bản đồ năng lượng dải tần từ DCT cục bộ trên block 8 × 8.']],[4,1.5,11.1])
p('Cách kết hợp hiện tại là ghép kênh đầu vào. Đây chưa phải mô hình có hai encoder riêng hoặc một module fusion riêng. Residual có thể chứa cạnh và texture; không đồng nhất với dấu vết cảm biến PRNU. DCT được dùng để phân tích tần số, không phải để nhúng watermark.')
sub('Vị trí của mô hình trong đề tài')
p('U-Net–ResNet34 là kiến trúc CNN, không phải SegFormer. ResNet34 được tích hợp vào vai trò encoder của U-Net. Việc đáp ứng yêu cầu tích hợp trong môn học cần được thống nhất với giảng viên. Chuyển toàn bộ sang SegFormer là thay kiến trúc, không phải tự động tích hợp SegFormer với U-Net.')

page(); h('3 Quy trình hoạt động từ ảnh đến mask')
table(['Bước','Xử lý','Kết quả'],[['1','Đọc ảnh và resize','Ảnh RGB kích thước 256 × 256.'],['2','Chuẩn bị đặc trưng','Đầu vào 3 hoặc 6 kênh, đúng công thức và chuẩn hóa khi training.'],['3','ResNet34 encoder','Feature maps đa mức độ phân giải.'],['4','U-Net decoder và skip connections','Feature maps được khôi phục độ phân giải.'],['5','Segmentation head','Bản đồ logits một kênh, 256 × 256.'],['6','Sigmoid','Điểm số trong khoảng 0–1 cho mỗi pixel.'],['7','Threshold','Mask nhị phân vùng nghi giả và vùng nền.']],[1.4,6,9.2])
sub('Sau decoder diễn ra điều gì')
p('Decoder chưa trực tiếp tạo ảnh trắng đen. Segmentation head tạo logits, tức điểm số thô có thể âm hoặc dương. Sigmoid chuyển chúng về khoảng 0–1. Ví dụ, logits −2,0; 0,4; 2,2 tương ứng gần đúng với 0,12; 0,60; 0,90. Với threshold 0,5, ba pixel này nhận nhãn 0; 1; 1.')
p('Giá trị 1 biểu thị vùng nghi giả; giá trị 0 biểu thị vùng nền. Điểm số sigmoid không nên được diễn giải thành độ tin cậy đã hiệu chỉnh của cả ảnh. Mask rỗng không chứng minh ảnh là thật.')
sub('Hậu xử lý phục vụ người dùng')
p('Đưa dự đoán về kích thước ảnh gốc theo quy tắc nội suy cố định để tạo overlay. Từ mask, xác định các thành phần liên thông để lấy bounding boxes và tính diện tích dự đoán bằng số pixel foreground chia tổng số pixel. Nếu nội suy mask nhị phân, dùng nearest neighbor; nếu nội suy bản đồ điểm số, áp dụng threshold sau nội suy và ghi rõ protocol.')
sub('Đầu ra trực tiếp và đầu ra cần bổ sung')
p('Đầu ra trực tiếp là bản đồ điểm số theo pixel. Mask được tạo bằng threshold. Box và diện tích là hậu xử lý; Real/Fake cấp ảnh cần quy tắc tổng hợp điểm số hoặc classification head được đánh giá riêng. IoU và Dice chỉ tính được khi ảnh có ground-truth mask.')

page(); h('4 Huấn luyện và ý nghĩa của seed')
p('Trong training, mô hình nhận ảnh và học bằng cách so sánh dự đoán với ground-truth mask. Loss được dùng để tính gradient; optimizer cập nhật trọng số. Khi inference, mô hình chỉ nhận ảnh, sử dụng trọng số đã học và không cập nhật chúng.')
p('Luồng training: ảnh → model → logits → BCE kết hợp Dice loss so với mask → backpropagation → cập nhật trọng số. Không threshold dự đoán trước khi tính loss. BCEWithLogits sử dụng logits; phần Dice sử dụng biểu diễn xác suất phù hợp.')
table(['Thông số','Thiết lập ghi nhận trong protocol'],[['Dataset hiện tại','CASIA v2 và mask theo pipeline dự án'],['Train và validation','Log ghi nhận 8.765 ảnh train và 1.768 ảnh validation'],['Kích thước và batch size','256 × 256; batch size 8'],['Learning rate và ngân sách','0,0001; 10 epoch cho mỗi lượt trong protocol có kiểm soát'],['Loss','Weighted BCE kết hợp Dice loss'],['Checkpoint selection','Validation micro Dice tại threshold 0,5; hòa thì chọn validation loss thấp hơn'],['Seed','7, 21 và 42; ba cấu hình tương ứng chín lượt training']],[5,11.6])
sub('Seed 7 21 và 42 là gì')
p('Seed là giá trị thiết lập bộ sinh số ngẫu nhiên. Nó ảnh hưởng đến các thao tác như khởi tạo trọng số chưa pretrained, thứ tự lấy mẫu và augmentation. Các số 7, 21, 42 không có ý nghĩa đặc biệt và không phải số epoch hoặc số ảnh.')
p('Chạy nhiều seed giúp kiểm tra độ ổn định của kết quả. Giữ nguyên train/validation/test khi so sánh các seed trong nghiên cứu này. Báo cáo từng lượt và trung bình ± độ lệch chuẩn mẫu; không chỉ chọn seed có điểm cao nhất. Tái lập còn phụ thuộc thư viện, phần cứng và cấu hình deterministic.')
p('Ngân sách 10 epoch cần được kiểm tra bằng learning curve. Nếu tăng ngân sách, áp dụng nhất quán cho các cấu hình và ghi nhận thành protocol rõ ràng trước khi đánh giá test.')

page(); h('5 Tiến độ và các khoảng trống')
table(['Hạng mục','Trạng thái ghi nhận'],[['U-Net baseline','Đã triển khai; có các thử nghiệm chẩn đoán.'],['U-Net–ResNet34 RGB Noise DCT','Đã triển khai ba biến thể và có phần training.'],['Checkpoint và resume','Có cơ chế lưu và tiếp tục training trong notebook.'],['Seed 7 và 21','Log trong PDF báo hoàn thành sáu lượt của ba cấu hình.'],['Seed 42 và tổng hợp chín lượt','Có code bổ sung; chưa xác nhận đầy đủ đầu ra.'],['CASIA test','Phần cuối PDF bị ngắt giữa cell; chưa có bảng test hoàn chỉnh để xác nhận.'],['Cross-dataset','Chưa có kết quả được xác nhận.'],['Demo đầy đủ đầu ra','Cần kiểm tra và hoàn thiện pipeline inference thống nhất.']],[6,10.6])
sub('Những kết luận chưa thể đưa ra')
p('Chưa đủ bằng chứng để xác định RGB, Noise hay DCT là phương án tốt nhất. Việc chạy được trên một ảnh mới chưa chứng minh dự đoán đáng tin cậy trên ảnh bất kỳ. Các checkpoint trên Drive cần được kiểm kê cùng config và history trước khi công nhận lượt chạy hoàn tất.')
sub('Rủi ro ảnh hưởng đến kết quả')
for s in ['Data leakage: ảnh gốc, donor, ảnh chỉnh sửa và ảnh trùng có thể liên quan giữa các split; kiểm tra hash chưa đủ để xác định toàn bộ quan hệ nguồn.', 'Mask không đúng cặp, sai polarity hoặc annotation không đáng tin cậy có thể làm sai cả training và evaluation.', 'Resize về 256 × 256 có thể làm mất chi tiết ở vùng giả nhỏ. Các cấu hình phải dùng preprocessing và protocol so sánh được.', 'Chọn checkpoint hoặc threshold bằng test sẽ làm mất tính độc lập của đánh giá.']: p(s,'List Bullet')

page(); h('6 Đánh giá và mục tiêu hoàn thành')
sub('Các chỉ số cần báo cáo')
p('Với vùng giả là lớp positive: TP là pixel giả dự đoán đúng, FP là pixel thật bị báo giả và FN là pixel giả bị bỏ sót. IoU = TP / (TP + FP + FN). Dice = 2TP / (2TP + FP + FN). Dice và Pixel-F1 trùng nhau khi dùng cùng định nghĩa và cách tổng hợp.')
p('Báo cáo micro IoU, micro Dice, Dice trung bình trên ảnh giả và tỷ lệ pixel báo giả trên ảnh thật. Phân biệt gộp pixel toàn tập với trung bình từng ảnh; ghi rõ cách xử lý mask rỗng. Với Real/Fake cấp ảnh, bổ sung confusion matrix, precision, recall và F1 trên tập có đủ lớp.')
sub('Roadmap theo thứ tự ưu tiên')
table(['Ưu tiên','Công việc','Điều kiện hoàn thành'],[['1','Kiểm kê các lượt chạy','Đủ bảng trạng thái config, history và checkpoint của chín lượt.'],['2','Khóa dữ liệu và protocol','Kiểm tra mask, split, leakage và quy tắc chọn checkpoint/ngưỡng.'],['3','Hoàn tất training','Các lượt còn thiếu hoàn tất; có learning curves và bảng validation.'],['4','CASIA test','Bảng từng seed, mean ± SD và kết quả từng ảnh.'],['5','Cross-dataset','Ít nhất một bộ ngoài CASIA; giữ checkpoint và ngưỡng, không fine-tune.'],['6','Demo và report','Mask, overlay, box, diện tích, image-level score và phân tích lỗi.']],[1.7,5.1,9.8])
p('Lộ trình đề xuất khoảng ba tuần, tùy GPU: tuần đầu kiểm kê và hoàn tất training; tuần hai đánh giá test và cross-dataset; tuần ba hoàn thiện demo, phân tích lỗi, báo cáo và slide. Đây là kế hoạch dự kiến, không phải thời gian chạy đã đo.')
sub('Goal tiền luận văn')
p('Hoàn thành một nghiên cứu tái lập được về ảnh hưởng của Noise và DCT đối với U-Net–ResNet34. Sản phẩm gồm bảng số có kiểm chứng, 12–20 ví dụ tốt và thất bại được chọn theo tiêu chí rõ ràng, giải thích giới hạn và pipeline dự đoán một ảnh. Không bảo đảm trước điểm số học phần hoặc mức chính xác.')

page(); h('7 Kiến thức cần nắm và tài liệu tham khảo')
sub('Thứ tự tìm hiểu dự án')
table(['Thứ tự','Nội dung','Tự kiểm tra bằng thực hành'],[['1','Detection và localization','Giải thích một ảnh, ground-truth mask và predicted mask.'],['2','Dữ liệu và split','Giải thích vì sao ảnh gốc và bản chỉnh sửa cần kiểm soát quan hệ nguồn.'],['3','U-Net và ResNet34','Vẽ encoder, decoder, skip connections và kích thước đầu vào/đầu ra.'],['4','Logits sigmoid và loss','Giải thích sự khác nhau giữa training và inference.'],['5','Noise và DCT','Hiển thị các bản đồ đặc trưng của cùng một ảnh.'],['6','Metrics và seed','Tính IoU/Dice trên mask nhỏ; đọc bảng mean ± SD.'],['7','Generalization','Giải thích vì sao CASIA test tốt chưa đủ cho mọi ảnh ngoài thực tế.']],[1.7,5.1,9.8])
sub('Tài liệu dự án')
p('[1] Notebook v4 - Colab3.pdf, bản được cung cấp để rà soát tiến độ. Các mốc tham chiếu: trang 292–293 mô tả protocol; trang 304 ghi log seed 7 và 21; trang 306–311 chứa code bổ sung seed 42 và tổng hợp; trang 312–314 bắt đầu phần CASIA test.')
sub('Tài liệu đọc thêm')
p('[2] Segmentation Models PyTorch. Quick Start. Ví dụ cấu hình U-Net với encoder ResNet34. https://segmentation-models-pytorch.readthedocs.io/en/latest/quickstart.html')
p('[3] Xie và cộng sự. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers, 2021. Dùng để đối chiếu kiến trúc; không phải mô hình hiện tại. https://arxiv.org/abs/2105.15203')
sub('Kết luận')
p('Dự án đã có kiến trúc U-Net–ResNet34 và ba biến thể đầu vào phục vụ nghiên cứu. Công việc quan trọng tiếp theo là chuyển các lượt training thành bằng chứng đánh giá độc lập, ổn định và có thể tái lập. Việc mở rộng sang Transformer hoặc fusion mới chỉ nên thực hiện sau khi bộ kết quả hiện tại hoàn chỉnh hoặc khi có yêu cầu cụ thể của giảng viên.')

out=Path('outputs/report_word/Bao_cao_UNet_ResNet34.docx')
doc.core_properties.title='Báo cáo tiến độ định vị vùng giả mạo ảnh bằng U-Net ResNet34'
doc.core_properties.subject='Mô hình, quy trình hoạt động, tiến độ và mục tiêu tiền luận văn'
doc.save(out)
print(out.resolve())
