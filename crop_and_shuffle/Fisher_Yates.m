function index_all = Fisher_Yates(len, iters)

index_all = [1:len];
for all = 1:iters
    for m = len:-1:2
        n = randi([1,m]);
        tmp = index_all(m);
        index_all(m) = index_all(n);
        index_all(n) = tmp;
    end
end