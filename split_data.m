files = dir('Corn_(maize)___Northern_Leaf_Blight');
files(1) = [];
files(1) = [];
[N, ~] = size(files);
tf = randperm(N) > (0.70 *N);

for idx = 1:N

    source = files(idx).folder+"/"+files(idx).name;
    if tf(idx) == 0
        dest = "train/Corn_(maize)___Northern_Leaf_Blight/";
    else
        dest = "test/Corn_(maize)___Northern_Leaf_Blight/";
    end
    movefile(source, dest);
end

files = dir('train/Corn_(maize)___Northern_Leaf_Blight');
files(1) = [];
files(1) = [];
[N, ~] = size(files);
tf = randperm(N) > (0.90 *N);

for idx = 1:N

    source = files(idx).folder+"/"+files(idx).name;
    if tf(idx) == 1
        dest = "val/Corn_(maize)___Northern_Leaf_Blight/";
        movefile(source, dest);
    end
    
end
