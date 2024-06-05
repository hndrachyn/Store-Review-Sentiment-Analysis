### Pendahuluan

#### Latar Belakang
Dalam era digital, perusahaan tidak hanya berinteraksi dengan pelanggan tetapi juga dihadapkan pada ulasan dan umpan balik yang diberikan pelanggan melalui berbagai platform. Ulasan ini dapat mempengaruhi persepsi publik terhadap perusahaan. Oleh karena itu, penting bagi perusahaan untuk memantau, mengamati, dan merespons ulasan pelanggan guna meningkatkan kinerja dan pelayanan.

#### Rumusan Masalah
1. Pengaruh ulasan pelanggan terhadap perusahaan.
2. Strategi perusahaan berdasarkan data ulasan pelanggan.
3. Respons perusahaan terhadap ulasan pelanggan.
4. Fenomena kata yang tidak ada di model otomatis menjadi sangat positif.

#### Tujuan dan Manfaat
Tujuan projek ini adalah membantu perusahaan menganalisis sentiment pelanggan untuk meningkatkan pelayanan dan mengatur strategi bisnis. Manfaatnya termasuk pemahaman terhadap preferensi pelanggan, peningkatan pelayanan, dan pengembangan strategi bisnis.

### Metode

#### Preprocessing
1. **Remove Punctuation:** Menghapus tanda baca.
2. **Lower Case:** Mengubah huruf menjadi huruf kecil.
3. **Remove Stopwords:** Menghapus kata-kata tidak penting.
4. **Lemmatizer:** Mengelompokkan kata ke bentuk kamus.

#### Ekstraksi Fitur
1. **TF-IDF:** Mentransformasi data teks menjadi data numerik dan memperhitungkan pentingnya kata dalam dokumen.
2. **Skip N-Gram:** Teknik untuk rangkaian kata dengan memungkinkan token dilewati.
3. **Naïve Bayes:** Metode klasifikasi berbasis probabilitas.
4. **SVM (Support Vector Machine):** Algoritma untuk menemukan hyperplane terbaik yang memisahkan dua kelas pada input space.