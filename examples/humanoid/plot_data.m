% plotting script

clc; clearvars;
close all;





myData = readtable('../../centroidalMPCDataWithoutStabilityCstr.txt');


left_steps = {Step([0;0.07;0], [0,0,0], 0, 1), ...
             Step([0.25;0.07;0], [0,0,0], 2, 5), ...
             Step([0.6;0.07;0], [0,0,0], 6, 9)};

right_steps  = {Step([0;-0.07;0], [0,0,0], 0, 3), ...
             Step([0.35;-0.07;0.0], [0,0,0], 4, 7), ...
             Step([0.6;-0.07;0], [0,0,0], 8, 9)};
nominal_left_steps = left_steps;
nominal_right_steps = right_steps;

% getting data from robot

dt                   = 0.01; % WB Block frequency
CoM                  = [myData.com_x myData.com_y myData.com_z];
CoM_d                = [myData.des_com_x myData.des_com_y myData.des_com_z];
% nominal_left_steps   = [myData.left_foot_next_pos_x myData.left_foot_next_pos_y myData.left_foot_next_pos_z];
% nominal_right_steps  = [myData.right_foot_next_pos_x myData.right_foot_next_pos_y myData.right_foot_next_pos_z];
measured_left_steps  = [myData.left_foot_pos_x myData.left_foot_pos_y myData.left_foot_pos_z];
measured_right_steps = [myData.right_foot_pos_x myData.right_foot_pos_y myData.right_foot_pos_z];
lforce_1 = [myData.left_foot_0_x myData.left_foot_0_y myData.left_foot_0_z];
lforce_2 = [myData.left_foot_1_x myData.left_foot_1_y myData.left_foot_1_z];
lforce_3 = [myData.left_foot_2_x myData.left_foot_2_y myData.left_foot_2_z];
lforce_4 = [myData.left_foot_3_x myData.left_foot_3_y myData.left_foot_3_z];

rforce_1 = [myData.right_foot_0_x myData.right_foot_0_y myData.right_foot_0_z];
rforce_2 = [myData.right_foot_1_x myData.right_foot_1_y myData.right_foot_1_z];
rforce_3 = [myData.right_foot_2_x myData.right_foot_2_y myData.right_foot_2_z];
rforce_4 = [myData.right_foot_3_x myData.right_foot_3_y myData.right_foot_3_z];


lforce   = [lforce_1';lforce_2';lforce_3';lforce_4'];
rforce   = [rforce_1';rforce_2';rforce_3';rforce_4'];


lforce_z = [lforce(3,:) ; lforce(6,:); lforce(9,:) ;lforce(12,:)];
rforce_z = [rforce(3,:) ; rforce(6,:); rforce(9,:); rforce(12,:)];

angMom   = [myData.ang_x myData.ang_y myData.ang_z];




%% PLOT FOOTPATHS
% plot left footpath


set(0,'DefaultLegendAutoUpdate','off')
filename = 'CentroidalMPCTestNoStability.gif';
h = figure('Renderer', 'painters');
title("without and with stability constraints");
hold on; grid on;

xlim([-0.2  1.0]);
ylim([-0.3  0.3]);

t     = 0:dt:length(CoM)*dt;

left_      = plot_footpath(nominal_left_steps{1}.position' , 0,'#3da4ab');
right_     = plot_footpath(nominal_right_steps{1}.position' , 0,'#3da4ab');
l_adapted_ = plot_footpath(measured_left_steps(1,:)' , 0,'#FFA500');
r_adapted_ = plot_footpath(measured_right_steps(1,:)' , 0,'#FFA500');

anim = animatedline('Color', '#CB7266', 'LineWidth',2);
frame = getframe(h);
im = frame2im(frame);
[imind,cm] = rgb2ind(im,256);

com = CoM;
addpoints(anim, com(1, 1), com(1, 2));

ylabel("$y [m]$", "FontSize", 20, Interpreter="latex");
xlabel("$x [m]$", "FontSize", 20, Interpreter="latex");
legend([left_, l_adapted_, anim], {'Nominal', 'Adapted', 'CoM'});
drawnow;
imwrite(imind,cm,filename,'gif', 'Loopcount',inf, 'DelayTime', 0.1);

for i = 1:length(nominal_right_steps)
    left_      = plot_footpath(nominal_left_steps{i}.position, 0 ,'#3da4ab');
    right_     = plot_footpath(nominal_right_steps{i}.position , 0,'#3da4ab');
end


for i=1:length(CoM)
    if (min(lforce_z(:,i)) > 1)
        l_adapted_ = plot_footpath(measured_left_steps(i,:) , 0,'#FFA500');
    elseif (min(rforce_z(:,i)) > 1)
        r_adapted_ = plot_footpath(measured_right_steps(i,:)' , 0,'#FFA500');
    end

    addpoints(anim, com(i, 1), com(i, 2));

    drawnow;
    frame = getframe(h);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);

    imwrite(imind,cm,filename,'gif','WriteMode','append', 'DelayTime', 0.1);
end





clc; clearvars;
close all;

myData = readtable('../../centroidalMPCDataWithStabilityCstr.txt');


left_steps = {Step([0;0.07;0], [0,0,0], 0, 1), ...
             Step([0.25;0.07;0], [0,0,0], 2, 5), ...
             Step([0.6;0.07;0], [0,0,0], 6, 9)};

right_steps  = {Step([0;-0.07;0], [0,0,0], 0, 3), ...
             Step([0.35;-0.07;0.0], [0,0,0], 4, 7), ...
             Step([0.6;-0.07;0], [0,0,0], 8, 9)};
nominal_left_steps = left_steps;
nominal_right_steps = right_steps;

% getting data from robot

dt                   = 0.01; % WB Block frequency
CoM                  = [myData.com_x myData.com_y myData.com_z];
CoM_d                = [myData.des_com_x myData.des_com_y myData.des_com_z];
% nominal_left_steps   = [myData.left_foot_next_pos_x myData.left_foot_next_pos_y myData.left_foot_next_pos_z];
% nominal_right_steps  = [myData.right_foot_next_pos_x myData.right_foot_next_pos_y myData.right_foot_next_pos_z];
measured_left_steps  = [myData.left_foot_pos_x myData.left_foot_pos_y myData.left_foot_pos_z];
measured_right_steps = [myData.right_foot_pos_x myData.right_foot_pos_y myData.right_foot_pos_z];
lforce_1 = [myData.left_foot_0_x myData.left_foot_0_y myData.left_foot_0_z];
lforce_2 = [myData.left_foot_1_x myData.left_foot_1_y myData.left_foot_1_z];
lforce_3 = [myData.left_foot_2_x myData.left_foot_2_y myData.left_foot_2_z];
lforce_4 = [myData.left_foot_3_x myData.left_foot_3_y myData.left_foot_3_z];

rforce_1 = [myData.right_foot_0_x myData.right_foot_0_y myData.right_foot_0_z];
rforce_2 = [myData.right_foot_1_x myData.right_foot_1_y myData.right_foot_1_z];
rforce_3 = [myData.right_foot_2_x myData.right_foot_2_y myData.right_foot_2_z];
rforce_4 = [myData.right_foot_3_x myData.right_foot_3_y myData.right_foot_3_z];


lforce   = [lforce_1';lforce_2';lforce_3';lforce_4'];
rforce   = [rforce_1';rforce_2';rforce_3';rforce_4'];


lforce_z = [lforce(3,:) ; lforce(6,:); lforce(9,:) ;lforce(12,:)];
rforce_z = [rforce(3,:) ; rforce(6,:); rforce(9,:); rforce(12,:)];

angMom   = [myData.ang_x myData.ang_y myData.ang_z];




%% PLOT FOOTPATHS
% plot left footpath


set(0,'DefaultLegendAutoUpdate','off')
filename = 'CentroidalMPCTest.gif';
h = figure('Renderer', 'painters');
title("without and with stability constraints");
hold on; grid on;

xlim([-0.2  1.0]);
ylim([-0.3  0.3]);

t     = 0:dt:length(CoM)*dt;

left_      = plot_footpath(nominal_left_steps{1}.position' , 0,'#3da4ab');
right_     = plot_footpath(nominal_right_steps{1}.position' , 0,'#3da4ab');
l_adapted_ = plot_footpath(measured_left_steps(1,:)' , 0,'#FFA500');
r_adapted_ = plot_footpath(measured_right_steps(1,:)' , 0,'#FFA500');

anim = animatedline('Color', '#CB7266', 'LineWidth',2);
frame = getframe(h);
im = frame2im(frame);
[imind,cm] = rgb2ind(im,256);

com = CoM;
addpoints(anim, com(1, 1), com(1, 2));

ylabel("$y [m]$", "FontSize", 20, Interpreter="latex");
xlabel("$x [m]$", "FontSize", 20, Interpreter="latex");
legend([left_, l_adapted_, anim], {'Nominal', 'Adapted', 'CoM'});
drawnow;
imwrite(imind,cm,filename,'gif', 'Loopcount',inf, 'DelayTime', 0.1);

for i = 1:length(nominal_right_steps)
    left_      = plot_footpath(nominal_left_steps{i}.position, 0 ,'#3da4ab');
    right_     = plot_footpath(nominal_right_steps{i}.position , 0,'#3da4ab');
end


for i=1:length(CoM)
    if (min(lforce_z(:,i)) > 1)
        l_adapted_ = plot_footpath(measured_left_steps(i,:) , 0,'#FFA500');
    elseif (min(rforce_z(:,i)) > 1)
        r_adapted_ = plot_footpath(measured_right_steps(i,:)' , 0,'#FFA500');
    end

    addpoints(anim, com(i, 1), com(i, 2));

    drawnow;
    frame = getframe(h);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);

    imwrite(imind,cm,filename,'gif','WriteMode','append', 'DelayTime', 0.1);
end



% t     = 0:dt:length(CoM)*dt;
% t(end) = [];
%
% figure('name','CoM on the plane')
% subplot(2,1,1);
% hold on; grid on;
% plot(t, CoM(:,1),  'LineWidth', 2);
% plot(t, CoM_d(:,1),  'LineWidth', 2);
% % plot(qas, qas_d, '--','color','#D95319', 'LineWidth', 2);
% l = xlabel('Time',  'FontSize', 30);
% set(l,'Interpreter','Latex');
% % l = ylabel('$x$ component of', 'FontSize', 30);
% set(l,'Interpreter','Latex');
% l = legend('Actual','Desired', 'FontSize', 30);
% set(l,'Interpreter','Latex');
%
%
% subplot(2,1,2);
% hold on; grid on;
% plot(t, CoM(:,2),  'LineWidth', 2);
% plot(t, CoM_d(:,2),  'LineWidth', 2);
% % plot(qas, qas_d, '--','color','#D95319', 'LineWidth', 2);
% l = xlabel('Time',  'FontSize', 30);
% set(l,'Interpreter','Latex');
% % l = ylabel('$y$ component', 'FontSize', 30);
% set(l,'Interpreter','Latex');
% l = legend('Actual','Desired', 'FontSize', 30);
% set(l,'Interpreter','Latex');
%
%
%
% figure('name','CoM tracking error')
%
% hold on; grid on;
% plot(t, CoM(:,1) - CoM_d(:,1), 'LineWidth', 2);
% plot(t, CoM(:,2) - CoM_d(:,2),  'LineWidth', 2);
% l = xlabel('Time',  'FontSize', 30);
% set(l,'Interpreter','Latex');
% l = ylabel('$\|p^n_{CoM} - p_{CoM}\|$', 'FontSize', 30);
% set(l,'Interpreter','Latex');
% l = legend('$e_x$', '$e_y$','FontSize', 30);
% set(l,'Interpreter','Latex');
%
%
% figure('name','angular momentum')
%
% hold on; grid on;
% plot(t, angMom(:,1), 'LineWidth', 2);
% plot(t, angMom(:,2) ,  'LineWidth', 2);
% plot(t, angMom(:,3) ,  'LineWidth', 2);
% l = xlabel('Time',  'FontSize', 30);
% set(l,'Interpreter','Latex');
% l = ylabel('$h^w(t)$', 'FontSize', 30);
% set(l,'Interpreter','Latex');
% l = legend('$h^w_x$', '$h^w_y$','$h^w_z$' ,'FontSize', 30);
% set(l,'Interpreter','Latex');




function p = plot_footpath(pos, angle, color)

%% SET FOOT SIZE
x = [-0.08, -0.08, 0.08, 0.08];
y = [-0.03,  0.03, 0.03, -0.03];

%% PLOT FOOTPATHS
    % plot left footpath
    left_foot_transform = hgtransform;
    left_foot_transform.Matrix = makehgtform('translate', pos,...
                                             'zrotate', angle);
    p = patch('XData',x,'YData',y,'FaceColor',color,'Parent',left_foot_transform);
    alpha(p, 0.4);
end


function plot_aesthetic(Title, Label_x, Label_y, Label_z, varargin)
% PLOT_AESTHETIC add Title, label and legends in a plot
%   PLOT_AESTHETIC(Title, Label_x, Label_y, Label_z, Legend_1, ..., Legend_n)
%   add title, labels and legends in a plot. LaTex syntax is allowed.

% set labels
if ~isempty(Label_x)
    x_label = xlabel(Label_x);
    set(x_label, 'Interpreter', 'latex', 'FontSize', 25);
end

if ~isempty(Label_y)
    y_label = ylabel(Label_y);
    set(y_label,'Interpreter','latex', 'FontSize', 25);
end

if ~isempty(Label_z)
    z_label = zlabel(Label_z);
    set(z_label,'Interpreter','latex');
    set(z_label,'FontSize', 20);
end

% set legend
if ~isempty(varargin)
    % get the legend object
    leg = get(legend(gca),'String');

    % if the legend does not exist create a new one

    for i = 1:length(varargin)
        varargin(i) = strrep(varargin(i),'_',' ');
    end

    if isempty(leg)
        new_legend = varargin;
    else
        old_legend = leg;
        % when a new plot is draw an automatic string is added to the
        % legend
        new_legend = [old_legend(1:end-1), varargin{:}];
    end
      % h = legend(varargin, 'Location', 'northoutside', 'Orientation','horizontal');
    h = legend(varargin, 'Location', 'best');
    set(h,'Interpreter','latex')
    set(h,'FontSize', 16);
end

% change linewidth
h = findobj(gcf,'type','line');
set(h,'linewidth',2)

% set the title
if ~isempty(Title)
    tit = title(Title);
    set(tit,'FontSize', 20);
    set(tit,'Interpreter','latex');
end

% change font size
set(gca,'FontSize', 18)

% set grid
grid on;
end


