nvcc -g -maxrregcount 128 --resource-usage -lineinfo -Xptxas -lineinfo -v -O3 -arch=all \
        main.cu -m64 -o cuda \
        -DBOINC -Iboinc/ -Lboinc/lib/lin -lcuda -lboinc_api -lboinc -Xptxas -v
        ls -la