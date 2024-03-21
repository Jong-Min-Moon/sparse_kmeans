 n = 200;
 a = 0.1;
 delta = sqrt(2 * (1+a) * log(n));
 
 
 x = [-delta * ones(n/2, 1) ; delta * ones(n/2, 1)] + randn(n, 1);

 A = x * x';
 
 [Z, obd] = kmeans_sdp(A, 2);

imagesc(Z)
title('SDP solution')
colormap('winter')
colorbar
colormap(flipud(colormap));
set(gca,'xtick',[])
set(gca,'ytick',[])
axis equal
a=axis;
a(1)=a(3);
a(2)=a(4);
axis(a)

figure;
imagesc(A)
title('Affinity matrix')
colormap('winter')
colorbar
colormap(flipud(colormap));
set(gca,'xtick',[])
set(gca,'ytick',[])
axis equal
a=axis;
a(1)=a(3);
a(2)=a(4);
axis(a)

 Z = kmeans_sdp(A.*(abs(A)>7), 2);

figure;
imagesc(Z)
title('SDP solution (absolute value)')
colormap('winter')
colorbar
colormap(flipud(colormap));
set(gca,'xtick',[])
set(gca,'ytick',[])
axis equal
a=axis;
a(1)=a(3);
a(2)=a(4);
axis(a)

figure;
imagesc(abs(A))
title('Affinity matrix (absolute value)')
colormap('winter')
colorbar
colormap(flipud(colormap));
set(gca,'xtick',[])
set(gca,'ytick',[])
axis equal
a=axis;
a(1)=a(3);
a(2)=a(4);
axis(a)









