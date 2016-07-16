import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
from bokeh.plotting import *
from bokeh.models import ColumnDataSource
import numpy as np
import scipy.io as sio
import os

def plot_results(values, pmode, p, pll, pul, plot_raster = False, xlim = None, ylim = [-0.5,1.5]):
    '''
    Visualize the results EM
    '''
    fig = plt.figure(figsize = [15,5])
    ccc = 'b'
    line, = plt.plot(pmode,  linestyle = '-', color= 'b', alpha=0.9,lw=1,label='model probability')
    plt.fill_between(range(0,len(p)),pll,pul,color='blue',alpha=0.2)
    blue_patch = mpatches.Patch(color='blue', alpha = 0.4,label='Uncertainty')
    plt.legend(handles=[blue_patch,line])
    if xlim is None:
        plt.xlim([0,len(p)+1])
    else:
        plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.grid('off')
    plt.show()
    
    if(plot_raster):
        I, J = np.where(values != 0)
        plt.figure(figsize = [15,4])
        plt.scatter(J,I,marker='x',color='green',alpha = 0.6)
        plt.title('Raster Plot from Data')
        plt.xlabel('Time')
        plt.ylabel('Trials')
        plt.ylabel
        plt.show()

# Functions for plotting the autocorrelation between the Gibbs samples
def autocorr(x):
    x = np.copy(x)
    x -= np.mean(x)
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:] / np.linalg.norm(x)**2

# define a function to plot the autocorrelations of the samples
def plot_autocorr(w,x,xlim):
    plt.figure(figsize=[16,4])
    plt.plot(autocorr(w), 'b-',label = 'w')
    plt.plot(autocorr(x), 'r-',label = 'x')
    plt.xlim(xlim)
    plt.title('Autocorrelation plot for variables $w$ and $x$', fontsize = 14)
    plt.xlabel('Lag',fontsize=12)
    plt.ylabel('Aurocorrelation',fontsize=12)
    plt.legend(fontsize=14)
    plt.show()

##################################################### Functions for Plotting 2D Bayesian EM Results #################################################
# Function for plotting the results from EM
def bokeh_plot_results(y, pmode, p, pll, pul, plot_data = True, out_notebook = True, html = "", trials = False):
    '''
    Visualize the results EM
    '''
    if(out_notebook):
        output_notebook()
    else:
        output_file("%s.html"%(html))
    TOOLS = "resize,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"    
  
    xlabel = 'Time (ms)'
    if (trials): xlabel = 'Trials'
    fig = figure(tools = TOOLS, x_axis_label=xlabel, y_axis_label='Probability')
    x = np.arange(len(pmode))
    #y = y.reshape([-1])
    if plot_data:
        I, J = np.where(y != 0)
        fig.scatter(J,I)
    fig.line(x, pmode,  line_color= 'blue',legend='model probability')
    fig.line(x, pul,  line_color= 'red',legend='Uncertainty')
    fig.line(x, pll,  line_color= 'red')
    show(fig)
    
def plot_sigmas(sigmas):
    # Plot the trace for sigma2e during EM
    output_notebook()
    TOOLS = "resize,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"    

    fig = figure(tools = TOOLS, x_axis_label='Iterations', y_axis_label='Sigma2e')
    x = np.arange(len(sigmas))
    fig.line(x, sigmas,  line_color= 'blue',legend='sigma2e trace')
    show(fig)


##################################################### Functions for Plotting Stephen's Data #########################################################
def bokeh_plot_data(Y):
    num_trials = Y.shape[0]
    trial_len = Y.shape[1]
    # Population data
    I,J = np.where(Y != 0)
    psthY = np.sum(Y,axis = 0)/float(num_trials)
    output_notebook()
    TOOLS = "resize,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"   
    # Plot
    p0 = figure(tools = TOOLS, title="Population Raster", x_axis_label='Time (ms)', y_axis_label='A')
    p0.scatter(J, I, line_width=2)
    p1 = figure(tools = TOOLS, title="Population PSTH", x_axis_label='Time (ms)', y_axis_label='A', x_range=p0.x_range)
    p1.line(np.arange(0,trial_len), psthY, line_width=2)
    p = gridplot([[p0], [p1]])
    show(p)
    
def bokeh_plot_raster(filename,celnum,celletter,baseln,exprmnt,path = ""):
    if path == "":
        path = '/Users/zyzdiana/Dropbox/DianaDemba StatesSpace/'

    data = sio.loadmat(path + filename)
    neuron = 'sig%s%s'%(celnum,celletter)
    length = np.ceil(data[neuron][-1]*1000)
    y = np.zeros([length])
    indicies = np.floor(data[neuron]*1000).astype(int)
    y[indicies.reshape([1,-1])[0]] = 1
    tone = data['tone']
    endtone = tone + 20000
    bigend = np.hstack([tone,endtone]).T
    numtrials = len(tone)
    Y = np.zeros([numtrials,baseln+exprmnt])
    for i in xrange(numtrials):
        Y[i,:] = y[np.ceil(bigend[0,i]*1000)-baseln:np.ceil(bigend[0,i]*1000)+exprmnt]
    bokeh_plot_data(Y)

def get_s_population_data(filename,celnum,celletter,baseln,exprmnt,path = ""):
    if path == "":
        path = '/Users/zyzdiana/Dropbox/DianaDemba StatesSpace/'

    data = sio.loadmat(path + filename)
    neuron = 'sig%s%s'%(celnum,celletter)
    length = np.ceil(data[neuron][-1]*1000)
    y = np.zeros([length])
    indicies = np.floor(data[neuron]*1000).astype(int)
    y[indicies.reshape([1,-1])[0]] = 1
    tone = data['tone']
    endtone = tone + 20000
    bigend = np.hstack([tone,endtone]).T
    numtrials = len(tone)
    Y = np.zeros([numtrials,baseln+exprmnt])
    for i in xrange(numtrials):
        Y[i,:] = y[np.ceil(bigend[0,i]*1000)-baseln:np.ceil(bigend[0,i]*1000)+exprmnt]
    return Y

##################################################### Functions for Plotting Whisker Data #########################################################
def bokeh_plot_raw_data(dir_num, neuron_num, stim_num, flag = True, path = ""):
    if path == "":
        path = '/Users/zyzdiana/Dropbox/Bayesian_State_Space/WhiskerDataForDiana/Data/'

    filepath = os.path.join(path,'Dir%d/Neuron%d/' % (dir_num, neuron_num))
    trng_data = sio.loadmat(os.path.join(filepath,'Stim%d/trngdataBis' % stim_num))
    test_data = sio.loadmat(os.path.join(filepath,'Stim%d/testdataBis' % stim_num))

    t = np.hstack([trng_data['t'], test_data['t']])
    y = np.hstack([trng_data['y'], test_data['y']])
    trial_len = 3000
    num_trials = t.shape[1]/trial_len

    num_trg = 17
    num_test = num_trials - num_trg
    
    # Population data
    rastaY = np.reshape(y, (num_trials,trial_len))
    I,J = np.where(rastaY != 0)
    psthY = np.sum(rastaY,axis = 0)/float(num_trials)

    # Training data
    rastaYtrg = np.reshape(y[:,:trial_len*num_trg], (num_trg,trial_len))
    Itrg,Jtrg = np.where(rastaYtrg != 0)
    psthYtrg = np.sum(rastaYtrg,axis = 0)/float(num_trg)

    # Test data
    rastaYtest = np.reshape(y[:,trial_len*num_trg:], (num_test,trial_len))
    Itest,Jtest = np.where(rastaYtest!=0)
    psthYtest = np.sum(rastaYtest,axis = 0)/float(num_test)
    output_notebook()
    TOOLS = "resize,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"    
    if flag:
        # Population plots
        p0 = figure(tools = TOOLS, title="Stimulus", x_axis_label='Time (ms)', y_axis_label='A')
        p0.line(np.arange(0,trial_len), t[0][:trial_len], line_width=2)
        p1 = figure(tools = TOOLS, title="Population Raster", x_axis_label='Time (ms)', y_axis_label='A')
        p1.scatter(J, I, line_width=2)
        p2 = figure(tools = TOOLS, title="Population PSTH", x_axis_label='Time (ms)', y_axis_label='A')
        p2.line(np.arange(0,trial_len), psthY, line_width=2)
        
        # Training data plots
        p3 = figure(tools = TOOLS, title="Stimulus", x_axis_label='Time (ms)', y_axis_label='A')
        p3.line(np.arange(0,trial_len), t[0][:trial_len], color='green',line_width=2)
        p4 = figure(tools = TOOLS, title="Training Data Raster", x_axis_label='Time (ms)', y_axis_label='A')
        p4.scatter(Jtrg, Itrg, color='green',line_width=2)
        p5 = figure(tools = TOOLS, title="Training Data PSTH", x_axis_label='Time (ms)', y_axis_label='A')
        p5.line(np.arange(0,trial_len), psthYtrg, color='green',line_width=2)
        
        # Testing data plots
        p6 = figure(tools = TOOLS, title="Stimulus", x_axis_label='Time (ms)', y_axis_label='A')
        p6.line(np.arange(0,trial_len), t[0][:trial_len], color='darkred', line_width=2)
        p7 = figure(tools = TOOLS, title="Test Data Raster", x_axis_label='Time (ms)', y_axis_label='A')
        p7.scatter(Jtest, Itest, color='darkred', line_width=2)
        p8 = figure(tools = TOOLS, title="Test Data PSTH", x_axis_label='Time (ms)', y_axis_label='A')
        p8.line(np.arange(0,trial_len), psthYtest, color='darkred', line_width=2)
        p = gridplot([[p0], [p1], [p2], [p3], [p4], [p5], [p6], [p7], [p8]])
        show(p)
        
    else:
        p0 = figure(tools = TOOLS, title="Stimulus", x_axis_label='Time (ms)', y_axis_label='A')
        p0.line(np.arange(0,trial_len), t[0][:trial_len], line_width=2)
        p1 = figure(tools = TOOLS, title="Population Raster", x_axis_label='Time (ms)', y_axis_label='A', x_range=p0.x_range)
        p1.scatter(J, I, line_width=2)
        p2 = figure(tools = TOOLS, title="Population PSTH", x_axis_label='Time (ms)', y_axis_label='A', x_range=p0.x_range)
        p2.line(np.arange(0,trial_len), psthY, line_width=2)
        p = gridplot([[p0], [p1], [p2]])
        show(p)

def get_population_data(dir_num, neuron_num, stim_num, path = ""):    
    if path == "":
        path = '/Users/zyzdiana/Dropbox/Bayesian_State_Space/WhiskerDataForDiana/Data/'

    filepath = os.path.join(path,'Dir%d/Neuron%d/' % (dir_num, neuron_num))
    trng_data = sio.loadmat(os.path.join(filepath,'Stim%d/trngdataBis' % stim_num))
    test_data = sio.loadmat(os.path.join(filepath,'Stim%d/testdataBis' % stim_num))

    t = np.hstack([trng_data['t'], test_data['t']])
    y = np.hstack([trng_data['y'], test_data['y']])
    trial_len = 3000
    num_trials = t.shape[1]/trial_len

    num_trg = 17
    num_test = num_trials - num_trg

    # Population data
    rastaY = np.reshape(y, (num_trials,trial_len))
    I,J = np.where(rastaY != 0)
    psthY = np.sum(rastaY,axis = 0)/float(num_trials)
    return rastaY

def plot_raster(dir_num, neuron_num, stim_num, start_trial, end_trial, path = ""):

    if path == "":
        path = '/Users/zyzdiana/Dropbox/Bayesian_State_Space/WhiskerDataForDiana/Data/'

    filepath = os.path.join(path,'Dir%d/Neuron%d/' % (dir_num, neuron_num))
    trng_data = sio.loadmat(os.path.join(filepath,'Stim%d/trngdataBis' % stim_num))
    test_data = sio.loadmat(os.path.join(filepath,'Stim%d/testdataBis' % stim_num))

    t = np.hstack([trng_data['t'], test_data['t']])
    y = np.hstack([trng_data['y'], test_data['y']])
    trial_len = 3000
    num_trials = t.shape[1]/trial_len

    # Population data
    rastaY = np.reshape(y, (num_trials,trial_len))
    rastaY = rastaY[:,start_trial:end_trial]
    I,J = np.where(rastaY != 0)
    psthY = np.sum(rastaY,axis = 0)/float(num_trials)   

    color = sns.color_palette()[0]
    figsize = [10,9]
    fig, ax = plt.subplots(3, sharex=True)
    fig.set_figheight(figsize[1])
    fig.set_figwidth(figsize[0])
    ax[0].plot(np.arange(0,(end_trial-start_trial)),t[0][start_trial:end_trial],lw=2,color = color)
    ax[0].set_ylabel('A')
    ax[0].set_title('Stimulus')

    ax[1].scatter(J, I, s = 8)
    ax[1].set_ylabel('B')
    ax[1].set_title('Population Raster')

    ax[2].plot(np.arange(0,(end_trial-start_trial)),psthY,lw=2,color = color)
    ax[2].set_ylabel('C')
    ax[2].set_title('Population PSTH')
    ax[2].set_ylim([0,0.2])

    [ax[i].grid('off') for i in xrange(3)]
    plt.xlabel('Time (ms)')
    plt.xlim([start_trial-100,end_trial+100])
    plt.suptitle('Directory %s Neuron %s Stimulus %s' % (dir_num, neuron_num, stim_num), fontsize = 15)
    plt.show()
    plt.close(fig)    

def plot_raw_data(dir_num, neuron_num, stim_num, flag = True, xlim = [0,3000], figsize = [10,9], path = ""):
    if path == "":
        path = '/Users/zyzdiana/Dropbox/Bayesian_State_Space/WhiskerDataForDiana/Data/'
    filepath = os.path.join(path,'Dir%d/Neuron%d/' % (dir_num, neuron_num))
    trng_data = sio.loadmat(os.path.join(filepath,'Stim%d/trngdataBis' % stim_num))
    test_data = sio.loadmat(os.path.join(filepath,'Stim%d/testdataBis' % stim_num))

    t = np.hstack([trng_data['t'], test_data['t']])
    y = np.hstack([trng_data['y'], test_data['y']])
    trial_len = 3000

    num_trials = t.shape[1]/trial_len

    num_trg = 17
    num_test = num_trials - num_trg
    
    # Population data
    rastaY = np.reshape(y, (num_trials,trial_len))
    I,J = np.where(rastaY != 0)
    psthY = np.sum(rastaY,axis = 0)/float(num_trials)

    # Training data
    rastaYtrg = np.reshape(y[:,:trial_len*num_trg], (num_trg,trial_len))
    Itrg,Jtrg = np.where(rastaYtrg != 0)
    psthYtrg = np.sum(rastaYtrg,axis = 0)/float(num_trg)

    # Test data
    rastaYtest = np.reshape(y[:,trial_len*num_trg:], (num_test,trial_len))
    Itest,Jtest = np.where(rastaYtest!=0)
    psthYtest = np.sum(rastaYtest,axis = 0)/float(num_test)
    
    if flag:
        fig, ax = plt.subplots(3, sharex=True)
        fig.set_figheight(figsize[1])
        fig.set_figwidth(figsize[0])
        ax[0].plot(np.arange(0,trial_len),t[0][:trial_len],lw=2)
        ax[0].set_ylabel('A')
        ax[0].set_title('Stimulus')

        ax[1].scatter(J, I,s = 6, alpha = 0.8)
        ax[1].set_ylabel('B')
        ax[1].set_title('Population Raster')

        ax[2].plot(np.arange(0,trial_len),psthY,lw=2)
        ax[2].set_ylabel('C')
        ax[2].set_title('Population PSTH')

        [ax[i].grid('off') for i in xrange(3)]
        plt.xlabel('Time (ms)')
        plt.xlim(xlim)
        plt.show()
        plt.close(fig)
        
        # Training data plots
        color = sns.color_palette()[1]
        fig, ax = plt.subplots(3, sharex=True)
        fig.set_figheight(figsize[1])
        fig.set_figwidth(figsize[0])
        ax[0].plot(np.arange(0,trial_len),t[0][:trial_len],lw=2,color = color)
        ax[0].set_ylabel('A')
        ax[0].set_title('Stimulus')

        ax[1].scatter(Jtrg, Itrg, s = 8,color = color)
        ax[1].set_ylabel('B')
        ax[1].set_title('Training Data Raster')

        ax[2].plot(np.arange(0,trial_len),psthYtrg,lw=2,color = color)
        ax[2].set_ylabel('C')
        ax[2].set_title('Training Data PSTH')
        ax[2].set_ylim([0,0.2])

        [ax[i].grid('off') for i in xrange(3)]
        plt.xlabel('Time (ms)')
        plt.xlim(xlim)
        plt.show()
        plt.close(fig)
        
        # Testing data plots
        color = sns.color_palette()[2]
        fig, ax = plt.subplots(3, sharex=True)
        fig.set_figheight(figsize[1])
        fig.set_figwidth(figsize[0])
        ax[0].plot(np.arange(0,trial_len),t[0][:trial_len],lw=2,color = color)
        ax[0].set_ylabel('A')
        ax[0].set_title('Stimulus')

        ax[1].scatter(Jtest, Itest, s = 8,color = color)
        ax[1].set_ylabel('B')
        ax[1].set_title('Test Data Raster')

        ax[2].plot(np.arange(0,trial_len),psthYtest,lw=2,color = color)
        ax[2].set_ylabel('C')
        ax[2].set_title('Test Data PSTH')
        ax[2].set_ylim([0,0.2])

        [ax[i].grid('off') for i in xrange(3)]
        plt.xlabel('Time (ms)')
        plt.xlim(xlim)
        plt.show()
        plt.close(fig)
        
    else:
        color = sns.color_palette()[0]
        fig, ax = plt.subplots(3, sharex=True)
        fig.set_figheight(figsize[1])
        fig.set_figwidth(figsize[0])
        ax[0].plot(np.arange(0,trial_len),t[0][:trial_len],lw=2,color = color)
        ax[0].set_ylabel('A   (mm)')
        ax[0].set_title('Stimulus')

        ax[1].scatter(J, I, s = 8)
        ax[1].set_ylabel('B')
        ax[1].set_title('Population Raster')

        ax[2].plot(np.arange(0,trial_len),psthY,lw=2,color = color)
        ax[2].set_ylabel('C')
        ax[2].set_title('Population PSTH')
        #ax[2].set_ylim([0,0.2])

        [ax[i].grid('off') for i in xrange(3)]
        plt.xlabel('Time (ms)')
        plt.xlim(xlim)
        #plt.suptitle('Directory %s Neuron %s Stimulus %s' % (dir_num, neuron_num, stim_num), fontsize = 15)
        plt.show()
        plt.close(fig)