# CIFAR-10 Üzerinde Derin Öğrenme Modelleri

Bu proje, CIFAR-10 veri kümesini kullanarak derin öğrenme modelleri olan DenseNet, ResNet ve VGGNet'i eğitmeyi ve değerlendirmeyi amaçlar. CIFAR-10, 10 farklı sınıfa ait 60000 renkli görüntüden oluşan bir veri kümesidir.

## Kullanılan Modeller

- **DenseNet**: Yoğun bağlantılar kuran derin bir sinir ağı mimarisi.
- **ResNet**: Daha derin ağların eğitimini kolaylaştırmak için atlayan bağlantılar kullanan bir sinir ağı mimarisi.
- **VGGNet**: Derinlik ve karmaşıklık açısından orta seviyede bir mimariye sahip olan bir sinir ağı modeli.

## Eğitim

Her bir model için ayrı ayrı dosyalar kullanılmıştır:

- **DenseNet**: [densenet.py](https://github.com/Umit22/Cnn-Modeller/blob/main/denseNet.py)
- **ResNet**: [resnet.py](https://github.com/Umit22/Cnn-Modeller/blob/main/resNet.py)
- **VGGNet**: [vggnet.py](https://github.com/Umit22/Cnn-Modeller/blob/main/vggNet.py)

Her dosyada, ilgili modelin oluşturulması, derlenmesi, eğitilmesi ve değerlendirilmesi kodları bulunmaktadır. Her modelin mimarisi ve eğitim süreci ayrıntılı olarak açıklanmıştır.

## Kullanım

Projenin kullanımı için gereksinimler:

- Python 3.x
- TensorFlow

# Sonuçlar 
Her bir model, CIFAR-10 veri kümesi üzerinde eğitildikten sonra elde edilen doğruluk oranları şunlardır:

DenseNet model doğruluğu: (0.7213)
ResNet model doğruluğu: (0.6002)
VGGNet model doğruluğu: (0.1)

# Katkıda Bulunma 
Her türlü katkıya açığım. Lütfen işlem yapmak istediğiniz özellikleri veya hata düzeltmelerini içeren bir pull talebi gönderin.
