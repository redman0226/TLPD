for i in {1,2,3}
do
for j in {"aaa","bbb","ccc"}
do

a=./split_${j}_fold_${i}

echo $a

done
done 
