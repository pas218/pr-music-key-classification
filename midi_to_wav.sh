for midifile in POP909-Dataset/POP909/*; do
    if [ -d "$midifile" ]; then
        echo "$midifile"
        midifilenum="${midifile##*/}"
        echo "$midifilenum"

        for file in soundfonts/*; do
            if [[ "$file" == *.sf2 ]]; then
                sffilename="${file##*/}"
                sffilename="${sffilename%%.*}"
                # echo "$file"
                # echo "$sffilename"
                fluidsynth -n -i -F "sound_outputs/output_${midifilenum}_${sffilename}.wav" $file ./POP909-Dataset/POP909/$midifilenum/$midifilenum.mid
            fi
        done



    fi
        
done

# for file in soundfonts/*; do
#     sffilename="${file##*/}"
#     sffilename="${sffilename%%.*}"
#     # echo "$file"
#     # echo "$sffilename"
#     fluidsynth -n -i -F sound_outputs/output_$sffilename.wav $file ./POP909-Dataset/POP909/001/001.mid
# done

# fluidsynth -n -i -F sound_outputs/output.wav soundfonts/fzero.sf2 ./POP909-Dataset/POP909/001/001.mid