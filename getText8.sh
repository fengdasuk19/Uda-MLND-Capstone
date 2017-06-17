wget http://www.mattmahoney.net/dc/enwik9.zip
unzip enwik9.zip
perl wikifil.pl enwik9 > text8
truncate -s 100000000 text8
