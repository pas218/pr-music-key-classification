for file in soundfonts/*; do
    sffilename="${file##*/}"
    sffilename="${sffilename%%.*}"
    # echo "$file"
    # echo "$sffilename"
    fluidsynth -n -i -F sound_outputs/output_$sffilename.wav $file ./POP909-Dataset/POP909/001/001.mid
done

# fluidsynth -n -i -F sound_outputs/output.wav soundfonts/fzero.sf2 ./POP909-Dataset/POP909/001/001.mid