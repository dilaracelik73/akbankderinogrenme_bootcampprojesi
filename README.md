## 🌟 Akbank Derin Öğrenme Bootcamp: Yeni Nesil Proje Kampı

Bu proje, Akbank Derin Öğrenme Bootcamp kapsamında geliştirilmiştir.
Amaç, CNN (Convolutional Neural Network) mimarisi kullanarak derin öğrenme tabanlı bir görüntü sınıflandırma modeli geliştirmektir.

Katılımcılara şu alanlarda pratik deneyim kazandırılması hedeflenmiştir:
- Görüntü sınıflandırması
- Veri analizi ve görselleştirme
- Derin öğrenme modeli geliştirme
- Model değerlendirme ve yorumlama

## 🌸 Flower Classification with Deep Learning

Bu proje, **derin öğrenme tabanlı bir görüntü sınıflandırma** uygulamasıdır. Kaggle’daki **Flowers Recognition** veri seti kullanılarak, farklı çiçek türlerini ayırt eden bir model geliştirilmiştir.
Bu projenin amacı, derin öğrenme yöntemleri kullanarak görüntü sınıflandırma problemi çözmektir. Çiçeklerin farklı türlerini **(örneğin rose, daisy, sunflower vb.)** otomatik olarak tanıyabilen bir model geliştirilmiştir.
Bilgisayarla görme (computer vision) alanındaki bu çalışma sayesinde, görsellerden anlamlı bilgiler çıkarabilen sistemlerin nasıl kurulabileceği gösterilmektedir.

## 📚 Projenin Ana Yapısı Hakkında
### **Kullanılan veri seti:** Kaggle – Flowers Recognition Dataset
- **İçerik**: Veri setinde rose, daisy, sunflower, tulip, dandelion gibi farklı çiçek türlerine ait binlerce etiketli görsel bulunmaktadır.
- Görseller ilgili çiçek türüne göre ayrı klasörlerde tutulmuştur.
- Her görselin boyutu farklıdır, bu nedenle model eğitiminden önce yeniden boyutlandırma yapılmıştır.
- Veriler eğitim, doğrulama ve test kümelerine ayrılmıştır.
- **Veri Ayrımı:** %70 Eğitim (train) -  %15 Doğrulama (validation) -  %15 Test (test) verisi şeklinde yapılmıştır.

### **Kullanılan Yöntemler**
- Ortam kurulumu ve tekrarlanabilirlik için sabit seed değerleri atanmıştır.
- Veri Ön İşleme: Görseller belirli bir boyuta (ör. 224x224) yeniden ölçeklendirilmiştir.
- Görüntüler ImageDataGenerator ile yeniden ölçeklendirilmiş ve veri artırma (augmentation) uygulanmıştır.
 1. Döndürme (rotation)
 2. Yatay/ dikey çevirme (flip)
 3. Yakınlaştırma (zoom)
 4. Kaydırma (shift)

**Bu sayede modelin aşırı öğrenmesi (overfitting) engellenmiş ve genelleme kabiliyeti artmıştır.**

### **Kullanılan algoritmalar / modeller:**
- **CNN (Convolutional Neural Network):**
Konvolüsyon, aktivasyon (ReLU), havuzlama (max pooling) ve tam bağlı katmanlardan oluşan bir ağ tasarlanmıştır.
Çıkış katmanında softmax aktivasyonu kullanılarak çok sınıflı sınıflandırma yapılmıştır.
- **Transfer Öğrenme (Transfer Learning):**
Önceden ImageNet veri seti üzerinde eğitilmiş VGG16, ResNet50 gibi modeller kullanılmıştır.

**Bu modellerin alt katmanları sabitlenmiş (frozen), yalnızca üst katmanlar yeniden eğitilmiştir.
Böylece daha kısa sürede daha yüksek doğruluk elde edilmiştir.**

### **Eğitim sürecinde:**
- EarlyStopping ve ModelCheckpoint gibi callback fonksiyonları kullanılmıştır.
1. EarlyStopping: Modelin doğrulama kaybı iyileşmediğinde eğitimi durdurdu.
2. ModelCheckpoint: En iyi doğruluk değerine sahip ağırlıklar kaydedildi.
- Optimizasyon için Adam optimizer tercih edilmiştir.
- Eğitim sürecinde GPU hızlandırma kullanıldı.

### **Elde Edilen Sonuçlar**
- Eğitim süreci sonunda model yüksek doğruluk oranına ulaşmıştır. Başlangıçtan eğitilen modelde doğruluk oranı yaklaşık **%80–85** aralığında elde edilmiştir. **ResNet50** ile benzer şekilde yüksek doğruluk elde edilmiştir.
- Transfer öğrenme kullanılan modeller, sıfırdan eğitilen CNN’e göre daha yüksek başarı sağlamıştır. **VGG16 ile doğruluk %90+** seviyelerine ulaşmıştır.
- Tüm çalışmalar sonucunda, **çiçek türlerinin doğru sınıflandırılmasında %90+** doğruluk elde edilmiştir.

**Sonuç olarak**, transfer öğrenme yöntemleri sıfırdan eğitilen modellere göre daha iyi performans göstermiştir.
  Proje, derin öğrenmenin görüntü sınıflandırma problemlerinde **güçlü performans** gösterdiğini ortaya koymuştur.


## 🤔 Metrik Yorumu
Yukarıda da bahsettiğim gibi, birçok aşamadan geçirilen veri setindeki ana amacımız çiçekleri doğru şekilde tahmin edebilmekti.
1. İlk olarak CNN modeli ile eğitime başladım. Fakat sonuçlar ve doğruluk oranları beklediğim kadar yüksek çıkmadı. Buradan çıkardığım ders, modelin daha iyi tahminler yapabilmesi ve özellikle overfit olmaması için veri artırma (data augmentation) kullanmam gerektiğiydi.
2. Data augmentation sayesinde elimdeki çiçek görsellerini farklı açılardan modelime gösterebildim. Bu hem modelin daha fazla çeşitlilik görmesini sağladı hem de veri sayımı artırmış oldu. Eğer bu işlemleri yapmasaydım, model büyük ihtimalle eğitim verisini ezberleyip test verisinde düşük performans gösterecekti.
3. Sonraki aşamada transfer learning yöntemlerini denemeye karar verdim. Hazır modellerden VGG16 ve ResNet50’yi kullandım. Bu modeller zaten çok büyük veri setlerinde eğitilmiş oldukları için, sonuçlarım ciddi şekilde iyileşti. Özellikle küçük veri setlerinde transfer öğrenmenin çok avantajlı olduğunu net bir şekilde gördüm.
4. Tüm çalışmaların sonunda accuracy-loss grafiklerini, confusion matrix sonuçlarını ve tek görsel tahminlerini inceledim. Modelin yalnızca eğitim verisini ezberlemediğini, yeni görsellerde de doğru tahmin yapabildiğini gözlemledim. Bu da modelin gerçek hayatta da kullanılabilir olduğunu gösteriyor.
5. Tabii ki bazı zorluklarla da karşılaştım. Özellikle bazı çiçek türleri birbirine çok benzediğinden (örneğin daisy ve dandelion), model zaman zaman karıştırmalar yaptı. Bunun nedenini anlamak için Eigen-CAM ve Grad-CAM yöntemlerini kullandım. Böylece modelin görselin hangi bölgelerine dikkat ettiğini analiz edebildim.
6. Tüm bunlara ek olarak hiperparametrelerimde de oynamalar ve denemeler yaparak en iyi sonucu nasıl elde edebileceğimi araştırdım.
7. Learning rate, batch size, optimizer ve droput değerlerini deneyerek tablo oluşturdum ve kaydettim.
8. Bu denemeler sonucunda en iyi kombinasyon: Learning Rate: 0.001, Batch Size: 32, Dropout: 0.5, Optimizer: Adam olarak sonuçlandı.
9. İleride benzer veri setleri ile daha fazla veri artırımı yaparak daha da iyi sonuçlar elde edebileceğimi düşünüyorum. 

**Sonuç olarak, projenin başında belirlediğim amaca ulaştım. CNN’e ek olarak uyguladığım yöntemlerle modelimin doğruluk oranını %90’ın üzerine çıkarmayı başardım. Bu da çalışmamın başarılı olduğunu gösteriyor.**

## 🌱 Sonuç ve Gelecek Çalışmalar

Bu proje boyunca temel amacım çiçekleri yapay zekâ ile doğru bir şekilde sınıflandırmaktı. CNN ve transfer öğrenme yöntemleriyle çalışarak doğruluk oranını %90’ın üzerine çıkarabildim. 
Bu sonuç, yaptığım çalışmanın başarılı olduğunu ve gerçek dünyada da kullanılabilir bir model geliştirdiğimi gösteriyor.
Geleceğe yönelik olarak bu projeyi yalnızca model seviyesinde bırakmayı değil, daha da ileriye taşımayı düşünüyorum. 

Özellikle:
Mobil veya web tabanlı bir arayüz geliştirmek istiyorum. Büyük olasılıkla bir mobil uygulama olması daha uygun olur.
Kullanıcılar uygulamayı açıp çiçeğin fotoğrafını çektiğinde, model o çiçeğin türünü tanıyacak.
Sadece sınıflandırmakla kalmayıp, yapay zekâ destekli bir bilgi ekranı da eklenebilir. Örneğin:
- “Şu an bir papatya ile karşı karşıyasınız.”
- “Bu çiçeğin özellikleri şunlardır: …”
- “Çiçeğinizi daha sağlıklı yetiştirmek için şunlara dikkat etmelisiniz: …”
Böylece uygulama, kullanıcıya hem bilgi sağlayacak hem de bakım önerileri verecek.

İleride projeyi geliştirirken şunları da hayal ediyorum:
- Gerçek zamanlı veri toplama: Uygulama, farklı kullanıcıların yüklediği çiçek fotoğraflarını anonim şekilde toplayarak veri setini büyütebilir.
- Daha güçlü modeller: Yeni nesil yapay zekâ modelleri (örneğin Vision Transformers) ile doğruluğu artırmak mümkün.
- Kariyer yönlendirmesi: Böyle projeler yaparak mobil uygulama geliştirme, yapay zekâ entegrasyonu ve kullanıcı deneyimi (UX) gibi alanlarda kendimi geliştirmek istiyorum.

**Kısacası, bu proje benim için sadece bir ödev değil; gelecekte gerçek kullanıcıların işine yarayabilecek bir uygulamaya dönüşebilecek bir fikir oldu. 🌸**

## 📌Linkler
Aşağıda sizlere Kaggle Notebbok Linkimi ve Kaggle Dataset Linkimi bırakmaktayım.
- Notebook Linki: https://www.kaggle.com/code/dilaracelikdilara/akbankderinogrenme
- Dataset Linki: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
















  
