clear;
close all;

files = dir('monarch_closed/*.jpg');
filesClosed = dir('monarch_open/*.jpg');
Allfiles = dir('prueba/*.jpg');

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
        dist(i-1) = round(d/10);
        
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

net = feedforwardnet([6 5]);
%net = configure(net,coordenadas,salida);
net = train(net,coordenadas,salida);
A = round(net(coordenadas));

 figure();
 plot(1:length(salida),coordenadas,'o',1:length(salida),A,'*');

k = length(Allfiles);
allimagen = Allfiles(1:k,:);

for cont=1:k
    name = strcat('prueba/', allimagen(cont).name);
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
    coordenadas2(cont,1:features-1) = dist(1,1:features-1);
    coordenadas2(cont,features) = dist(features-1) - dist(1);
    
    figure(cont);
    imshow(I); hold on; plot(strongest1);
    
end

coordenadas2 = coordenadas2';

x = net(coordenadas2);
x = x';
x = round(x);

for i=1:k 
    if (x(i) ~= 1)
        x(i) = 0;
    end
end

casosExito = 0;

allImages = Allfiles(1:k,:);

for i=1:k
    name = strcat('prueba/', allImages(i).name);
    im = imread(name);
    I=rgb2gray(im);
    
    figure();
    if(x(i)==1) %si la imagen resultó clasificada como la clase 1%
    imshow(I); hold on; imshow('abierta.jpg');
    if (i <= 4) 
        casosExito = casosExito + 1;
    end
    elseif(x(i)==0) %si la imagen resultó clasificada como la clase 0%
    imshow(I); hold on; imshow('cerrada.jpg'); 
    if (i > 4) 
        casosExito = casosExito + 1;
    end
    end
  z = waitforbuttonpress;
end

porcentajeAciertos = round(casosExito/k*100);
