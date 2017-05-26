wget http://www.mattmahoney.net/dc/enwik9.zip
unzip enwik9.zip
perl wikifil_revision.pl enwik9 > text8_revision
truncate -s 100000000 text8_revision
