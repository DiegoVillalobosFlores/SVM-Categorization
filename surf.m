clear;
close all;

files = dir('monarch_open/*.jpg');
filesClosed = dir('monarch_closed/*.jpg');

features = 6;

n = length(files);
m = length(filesClosed);

images = files(1:n,:);

coordenadas = zeros(n+m,features*2);

for cont=1:n
    name = strcat('monarch_open/', images(cont).name);
    im = imread(name);
    I=rgb2gray(im);

    points1 = detectSURFFeatures(I);
    strongest1 = points1.selectStrongest(features); 
    [featuresSURF, valid_pointsSURF] = extractFeatures(I, strongest1);
    a=strongest1.Location;
    a = round(a/100);
    a(1:features,2) = sort(a(1:features,2));
    %a = sort(a);
    
    coordenadas(cont,1:features) = a(1:features,1);
    coordenadas(cont,features+1:features*2) = a(1:features,2);
    
end

images = filesClosed(1:m,:);

for cont=1:m
    name = strcat('monarch_closed/', images(cont).name);
    im = imread(name);
    I=rgb2gray(im);

    points1 = detectSURFFeatures(I);
    strongest1 = points1.selectStrongest(features); 
    [featuresSURF, valid_pointsSURF] = extractFeatures(I, strongest1);
    a=strongest1.Location;
    a = round(a/100);
    a(1:features,2) = sort(a(1:features,2));
    %a = sort(a,'descend');

    
    coordenadas(cont+n,1:features) = a(1:features,1);
    coordenadas(cont+n,features+1:features*2) = a(1:features,2);
   
end

salida(1:n,1) = 1;
salida(n+1:(n+m),1) = 0;

salida = salida';
coordenadas = coordenadas';

net = feedforwardnet([10 10]);
net = configure(net,coordenadas,salida);
net = train(net,coordenadas,salida);
A = net(coordenadas);

% figure();
% plot(1:length(salida),coordenadas,'o',1:length(salida),A,'*');

files = dir('todas/*.jpg');

k = length(files);
imagen = files(1:k,:);

features = 6;

for cont=1:k
    name = strcat('todas/', imagen(cont).name);
    im = imread(name);
    I=rgb2gray(im);

    points3 = detectSURFFeatures(I);
    strongest1 = points3.selectStrongest(features); 
    [featuresSURF, valid_pointsSURF] = extractFeatures(I, strongest1);
    b=strongest1.Location;
    b = round(b);
    %a = minmax(a);
    %a = sort(a);

    coordenadas2(cont,1:features) = b(1:features,1);
    coordenadas2(cont,features+1:features*2) = b(1:features,2);
end

x = net(coordenadas2');
x = x';
x = round(x);

d(1:m,1) = 0;
d(m+1:m+n,1) = 1;

promedio = 0;

for i=1:n+m
    if d(i,1) == x(i,1)
        promedio = promedio+1;
    end
end

promedio = round((promedio/(n+m))*100);

%subplot(1,2,2)
%imshow(I2); hold on; plot(strongest2);