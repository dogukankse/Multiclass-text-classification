# Multiclass Text Classification
 
 ### Özet

- Python 3.6 dili kullanılarak programlanmıştır.
- Gereksinimler ***requirements.txt*** adlı dosyanın içerisindedir.
- Gereksinimler dizinde `pip install -r requirements.txt` komutuyla indirilebilir. 
- Projede Zemberek-NLP 0.16 kullanılmıştır. ( Link: [Zemberek-NLP](https://github.com/ahmetaa/zemberek-nlp/) )
- Zemberek kütüphanesinin çalışması için en az [JDK8](https://www.oracle.com/technetwork/java/javase/downloads/index.html) gerekmektedir.
- Zemberek kütüphanesini kullanabilmek için gRPC kullanılmıştır.
- gRPC sunucusunun açmak için dizinde `java -jar zemberek-full.jar StartGrpcServer --dataRoot .\zemberek_data\` komutunun çalıştırılması gerekmektedir.
- Stop words Moodle'da verilenler kullanılmıştır.

### Gereksinimler
- Java 8
- Python 3.6
    - *grpcio==1.20.1*
    - *numpy==1.16.3*
    - *zemberek_grpc==0.16.1*
    - *nltk==3.4.1*
    - *pandas==0.24.2*
    - *grpc==0.3-19*
    - *scikit_learn==0.21.1*
