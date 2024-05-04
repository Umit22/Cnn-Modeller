# CIFAR-10 Üzerinde Derin Öğrenme Modelleri

Bu proje, CIFAR-10 veri kümesini kullanarak derin öğrenme modelleri olan DenseNet, ResNet ve VGGNet'i eğitmeyi ve değerlendirmeyi amaçlar. CIFAR-10, 10 farklı sınıfa ait 60000 renkli görüntüden oluşan bir veri kümesidir.

## Kullanılan Modeller

- **DenseNet**: Yoğun bağlantılar kuran derin bir sinir ağı mimarisi.
- **ResNet**: Daha derin ağların eğitimini kolaylaştırmak için atlayan bağlantılar kullanan bir sinir ağı mimarisi.
- **VGGNet**: Derinlik ve karmaşıklık açısından orta seviyede bir mimariye sahip olan bir sinir ağı modeli.

## Eğitim

Her bir model için ayrı ayrı dosyalar kullanılmıştır:

- **DenseNet**: [densenet.py](link)
- **ResNet**: [resnet.py](link)
- **VGGNet**: [vggnet.py](link)

Her dosyada, ilgili modelin oluşturulması, derlenmesi, eğitilmesi ve değerlendirilmesi kodları bulunmaktadır. Her modelin mimarisi ve eğitim süreci ayrıntılı olarak açıklanmıştır.

## Kullanım

Projenin kullanımı için gereksinimler:

- Python 3.x
- TensorFlow

Her bir modelin kodunu çalıştırmak için ilgili dosyayı kullanabilirsiniz:

```bash
python densenet.py
python resnet.py
python vggnet.py
