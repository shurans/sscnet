function show_volume(v, c)
v(v==255) =0;
if nargin == 1
    c = v+round(10*rand(1));
end

k = find(v);
[i1, i2, i3] = ind2sub(size(v), k);
scatter3(i1, i2, i3, 200, c(k), '.'); axis equal
colorbar

cls_names = {'empty','ceiling','floor','wall','window','door','chair','bed','sofa','table','bookshelf','cabinet','night_stand','bathtub','toilet','tv','obj'};

end