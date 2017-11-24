clear;
close all;

files = dir('monarch_open/*.jpg');
features = 6;


%Acomodar esta parte
imagen = imread('mno001.jpg');
imagen = rgb2gray(imagen);
imagenpoints = detectSURFFeatures(imagen);
strongest1 = imagenpoints.selectStrongest(features); 
[featuresSURF, valid_pointsSURF] = extractFeatures(imagen, strongest1);
imagen=strongest1.Location;
%

n = length(files);
images = files(1:n,:);

points = zeros(0:n);

for cont=1:n
    name = strcat('monarch_open/', images(cont).name);
    im = imread(name);
    I=rgb2gray(im);

    points1 = detectSURFFeatures(I);
    strongest1 = points1.selectStrongest(features); 
    [featuresSURF, valid_pointsSURF] = extractFeatures(I, strongest1);
    a=strongest1.Location;
    a = round(a);
    a = sort(a,'ascend');
     %a = minmax(a);
    
    points(cont).x = a(1:features,1);
    points(cont).y = a(1:features,2);
    
%     figure(1);
%     subplot(1,2,1)
%     imshow(I); hold on; plot(strongest1);
end

points = points';

cont2 = 1;
for cont=1:n
        for contF=1:features
            coordenadas(cont,contF) = points(cont).x(contF); 
            coordenadas(cont,contF+features) = points(cont).y(contF); 
            cont2 = cont2 +1;
        end
end


files = dir('monarch_closed/*.jpg');

m = length(files);
images = files(1:m,:);

features = 6;

points2 = zeros(0:m);

for cont=1:m
    name = strcat('monarch_closed/', images(cont).name);
    im = imread(name);
    I=rgb2gray(im);

    points1 = detectSURFFeatures(I);
    strongest1 = points1.selectStrongest(features); 
    [featuresSURF, valid_pointsSURF] = extractFeatures(I, strongest1);
     a=strongest1.Location;
     a = round(a);
     %a = minmax(a);
     a = sort(a,'descend');
    
    points2(cont).x = a(1:features,1);
    points2(cont).y = a(1:features,2);
    
%     figure(1);
%     subplot(1,2,1)
%     imshow(I); hold on; plot(strongest1);
end

cont2 = 1;
for cont=1:m
        for contF=1:features
            coordenadas(cont+n,contF) = points2(cont).x(contF); 
            coordenadas(cont+n,contF+features) = points2(cont).y(contF); 
            cont2 = cont2 +1;
        end
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

coordenadas2 = coordenadas(1,1);

img(1,1:features) = imagen(1:features,1);
img(1,features+1:features*2) = imagen(1:features,2);

img = round(img);
x = net(img');
x = x';
res = round(x);



% A = net(b);
% salida_1 = sim(net,b);

%subplot(1,2,2)
%imshow(I2); hold on; plot(strongest2);