for order in 0 1 2 3 4 5
do
	for pct_miss in 10 20 30 40 50
	do
		for att in 'T' 'F'
		do 
			for exp_num in {3..6}
			do
				s=$(printf "\-att %c -id 4 -e 4 --features \"[256,256,256,1]\" -lr 0.1 -wd 0.1 -k 2 -a relu -ns 0.1 --pct_miss %d --order %d --exp_num %d" $att $pct_miss $order $exp_num)
				s="${s:1}"
				a=$(date)
				echo $s > "$a".txt
				python test.py $s >> "$a".txt
			done
		done
	done
done
