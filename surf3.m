clear;
close all;

files = dir('monarch_closed/*.jpg');
filesClosed = dir('monarch_open/*.jpg');
Allfiles = dir('todas/*.jpg');

features = 5;

n = length(files);
m = length(filesClosed);

images = files(1:n,:);

coordenadas = zeros(n+m,features-1);

for cont=1:n
    name = strcat('monarch_closed/', images(cont).name);
    im = imread(name);
    I=rgb2gray(im);

    points1 = detectSURFFeatures(I);
    strongest1 = points1.selectStrongest(features); 
    [featuresSURF, valid_pointsSURF] = extractFeatures(I, strongest1);
    a=strongest1.Location;
    a = round(a);
    
    for i=2:features
        x1 = a(1,1);
        x2 = a(i,1);
        y1 = a(1,2);
        y2 = a(i,2);
        d = sqrt((x1-x2)^2 + (y1-y2)^2);
        dist(i-1) = round(d);
        
    end
    
    dist = sort(dist);
    
    coordenadas(cont,1:features-1) = dist(1,1:features-1);
    coordenadas(cont,features) = dist(features-1) - dist(1);
    
end

images = filesClosed(1:m,:);

for cont=1:m
    name = strcat('monarch_open/', images(cont).name);
    im = imread(name);
    I=rgb2gray(im);

    points1 = detectSURFFeatures(I);
    strongest1 = points1.selectStrongest(features); 
    [featuresSURF, valid_pointsSURF] = extractFeatures(I, strongest1);
    a=strongest1.Location;
    a = round(a);
    
    for i=2:features
        x1 = a(1,1);
        x2 = a(i,1);
        y1 = a(1,2);
        y2 = a(i,2);
        d = sqrt((x1-x2)^2 + (y1-y2)^2);
        dist(i-1) = sort(round(d/10));
    end
    
    dist = sort(dist);
    coordenadas(cont+n,1:features-1) = dist(1,1:features-1);
    coordenadas(cont+n,features) = dist(features-1) - dist(1);
    
end

salida(1:n,1) = 0;
salida(n+1:(n+m),1) = 1;

salida = salida';
coordenadas = coordenadas';

net = feedforwardnet([10 9 12]);
%net = configure(net,coordenadas,salida);
net = train(net,coordenadas,salida);
A = round(net(coordenadas));

presicionEntrenamiento = 0;

for i=1:158
    if (i<=n && A(i)==0)
        presicionEntrenamiento = presicionEntrenamiento +1;
    elseif (i>n && A(i)==1)
        presicionEntrenamiento = presicionEntrenamiento +1;
    end
end
        
presicionEntrenamiento = presicionEntrenamiento/158*100;

 figure();
 plot(1:length(salida),coordenadas,'o',1:length(salida),A,'*');

files = dir('todas/*.jpg');

k = length(files);
imagen = files(1:k,:);

x = net(coordenadas);
x = x';
x = round(x);

presicionPrueba = 0;

for i=1:158
    if (i<=n && x(i)==0)
        presicionPrueba = presicionPrueba +1;
    elseif (i>n && x(i)==1)
        presicionPrueba = presicionPrueba +1;
    end
end
        
presicionPrueba = presicionPrueba/158*100;


for i=1:k 
    if (x(i) ~= 1)
        x(i) = 0;
    end
end

casosExito = 0;

allImages = Allfiles(1:k,:);

for i=1:k
    name = strcat('todas/', allImages(i).name);
    im = imread(name);
    I=rgb2gray(im);
    
    %figure();
    if(x(i)==1) %si la imagen resultó clasificada como la clase 1%
    %imshow(I); hold on; imshow('abierta.jpg');
    if (i > n) 
        casosExito = casosExito + 1;
    end
    elseif(x(i)==0) %si la imagen resultó clasificada como la clase 0%
    %imshow(I); hold on; imshow('cerrada.jpg'); 
    if (i <= n) 
        casosExito = casosExito + 1;
    end
    end
  % z = waitforbuttonpress;
end

porcentajeAciertos = round(casosExito/k*100);
