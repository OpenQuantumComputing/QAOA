#!/bin/bash

problem_encoding="binary"

maxdepth=1
shots=100000


for casename in {"ErdosRenyi","BarabasiAlbert"}
#for casename in {"Barbell",}
do

    for k in {3,5,6,7}
    do

        #case full H

        if [ "$k" -eq 5 ] || [ "$k" -eq 6 ]; then
            clf_options=("LessThanK" "max_balanced")
        else
            clf_options=("LessThanK")
        fi

        for clf in "${clf_options[@]}"
        do

            for mixer in {"X","Grovertensorized"}
            do

                echo "fullH" $k $clf $mixer $casename $maxdepth $shots
                bash run_graphs.sh "fullH" $k $clf $mixer $casename $maxdepth $shots
            done
        done

        #case subspace

        for mixer in {"LX","Grover","Grovertensorized"}
        do

            echo "subH" $k "None" $mixer $casename $maxdepth $shots
            bash run_graphs.sh "subH" $k "None" $mixer $casename $maxdepth $shots
        done

        echo "------"

    done


done

