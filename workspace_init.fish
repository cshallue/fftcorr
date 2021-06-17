switch (uname)
    case Linux
        conda activate fftcorr
        export DYLD_FALLBACK_LIBRARY_PATH=/home/cshallue/miniconda3/envs/fftcorr/lib
    case Darwin
        conda_init fftcorr
        export DYLD_FALLBACK_LIBRARY_PATH=/Users/shallue/miniconda3/envs/fftcorr/lib
    case '*'
        echo "Unrecognized system!"
end
set --export "PYTHONPATH $HOME/git/fftcorr:$PYTHONPATH"
