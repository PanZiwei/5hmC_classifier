
Count the file number
Guppy v5.0.14 super model:
T4_lambda
cd /fastscratch/c-panz/2021-09-30/guppy5.0.14_supmodel_tombo/T4LambdaTF1.PerChrom
(base) [c-panz@sumner044 T4LambdaTF1.PerChrom]$ ls -1 J02459.1 | wc -l
5267

cd /fastscratch/c-panz/2021-09-30/guppy5.0.14_supmodel_tombo/T4LambdaTF1.PerChrom
find . \( -name "*.fast5" \) | wc -l
146832 #total fast5 numbers

Guppy v5.0.11 super model:
cd /pod/2/li-lab/Ziwei/Nanopore/data/single_read/T4LambdaTF1_guppy5.0.11_supmodel_tombo.PerChrom
ls -1 J02459.1 | wc -l
13155
ls -1 KJ477685.1 | wc -l
345


#5mC_lambmda
(base) [c-panz@sumner044 5mC_lambda]$ pwd
/pod/2/li-lab/Ziwei/Nanopore/data/single_read/guppy5.0.14.supmodel.tombo/5mC_lambda
(base) [c-panz@sumner044 5mC_lambda]$ ls | wc -l
14210

(base) [c-panz@sumner044 5mCLambdaTF1_guppy5.0.11_supmodel_tombo.PerChrom]$ pwd
/pod/2/li-lab/Ziwei/Nanopore/data/single_read/5mCLambdaTF1_guppy5.0.11_supmodel_tombo.PerChrom
(base) [c-panz@sumner044 5mCLambdaTF1_guppy5.0.11_supmodel_tombo.PerChrom]$ ls J02459.1/ | wc -l
26421