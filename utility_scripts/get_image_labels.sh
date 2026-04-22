#!/bin/bash


# avoid parsing ls
for i in {0..9}
  do
    for j in {0..9}
      do
        for k in {0..9}
        do
            soundfont="sonic2piano"
            filepath="./POP909-Dataset/POP909/${i}${j}${k}/key_audio.txt"
            echo "looking for file $filepath"
            
            if [ -f "$filepath" ]; then
                line=$(head -n 1 "$filepath")
                arr=($line)
                key="${arr[2]}"
                map_val="0"
                if [ "${arr[2]}" == "Ab:maj" ]; then
                    map_val="0"
                elif [ "${arr[2]}" == "A:maj" ]; then
                    map_val="1"
                elif [ "${arr[2]}" == "Bb:maj" ]; then
                    map_val="2"
                elif [ "${arr[2]}" == "B:maj" ]; then
                    map_val="3"
                elif [ "${arr[2]}" == "C:maj" ]; then
                    map_val="4"
                elif [ "${arr[2]}" == "Db:maj" ]; then
                    map_val="5"
                elif [ "${arr[2]}" == "D:maj" ]; then
                    map_val="6"
                elif [ "${arr[2]}" == "Eb:maj" ]; then
                    map_val="7"
                elif [ "${arr[2]}" == "E:maj" ]; then
                    map_val="8"
                elif [ "${arr[2]}" == "F:maj" ]; then
                    map_val="9"
                elif [ "${arr[2]}" == "Gb:maj" ]; then
                    map_val="10"
                elif [ "${arr[2]}" == "G:maj" ]; then
                    map_val="11"
                elif [ "${arr[2]}" == "Ab:min" ]; then
                    map_val="12"
                elif [ "${arr[2]}" == "A:min" ]; then
                    map_val="13"
                elif [ "${arr[2]}" == "Bb:min" ]; then
                    map_val="14"
                elif [ "${arr[2]}" == "B:min" ]; then
                    map_val="15"
                elif [ "${arr[2]}" == "C:min" ]; then
                    map_val="16"
                elif [ "${arr[2]}" == "Db:min" ]; then
                    map_val="17"
                elif [ "${arr[2]}" == "D:min" ]; then
                    map_val="18"
                elif [ "${arr[2]}" == "Eb:min" ]; then
                    map_val="19"
                elif [ "${arr[2]}" == "E:min" ]; then
                    map_val="20"
                elif [ "${arr[2]}" == "F:min" ]; then
                    map_val="21"
                elif [ "${arr[2]}" == "Gb:min" ]; then
                    map_val="22"
                elif [ "${arr[2]}" == "G:min" ]; then
                    map_val="23"
                fi
            echo "spect_${i}${j}${k}_$soundfont.png, $map_val" >> ./dataset/$soundfont/labels.csv
            fi
        done
    done
done
