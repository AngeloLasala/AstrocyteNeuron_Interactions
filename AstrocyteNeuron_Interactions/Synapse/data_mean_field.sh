#!/bin/sh
for i in 1 2.15 4.6 10 21 46 100  
	do
		python TM_model.py $i -p
done