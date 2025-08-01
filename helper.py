import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import pandas as pd

def plot_comparison(data, cluster_label, cluster_label2,
                    label, label2, agg = np.median, get_cluster = None,
                    get_sample = None,cluster_center = None, cluster_center2 = None,
                    get_feature = None,used_final_assignment = False,donomiate = None):

    plt.rcParams["figure.figsize"] = [20,20]

    plt.rcParams.update({'axes.labelsize': 15,
                        'axes.titlesize': 20,
                        'xtick.labelsize': 15,
                        'ytick.labelsize': 15})
    
    zip_name =[60088,60185,60411,60415,60607,60608,60609,60610,60611,60612,60613,60614,
                    60615,60616,60617,60618,60619,60620,60621,60622,60623,60625,60626,60628,
                    60629,60630,60631,60632,60633,60634,60636,60637,60638,60639,60640,60641,
                    60643,60644,60645,60646,60647,60649,60651,60652,60653,60654,60655,60656,
                    60657,60660]

    zip_name = [str(i) for i in zip_name]

    features = ['Full-service restaurants','Supermarkets and grocery stores',
                'Religious organizations','Child day care services','Hotels (except casino) and motels',
                'Beer, wine, and liquor stores']

    if get_cluster == None:
        get_cluster = [i for i in range(len(label.unique()))]
    if get_feature == None:
        print('get_feature is None')
        get_feature = [i for i in range(data.shape[1])]
    if get_sample == None:
        get_sample = zip_name
    color = ['#cc79a7','#0072b2','#d55e00']
    marker = ['s','o','^']
    seleted_cluster = [0,1,2]
    title = ['Full-service Restaurants','Supermarkets and Grocery Stores',
                'Religious Organizations','Child Day Care Services','Hotels (except Casino) and Motels',
                'Beer, Wine, and Liquor Stores']
    col = ['Full-service restaurants','Supermarkets and grocery stores',
                'Religious organizations','Child day care services','Hotels (except casino) and motels',
                'Beer, wine, and liquor stores']
    fig = plt.figure()
    gs = fig.add_gridspec(4, 3)
    axs = gs.subplots()
    count = 0
    for feature in get_feature:
        for i in range(len(col)):
            if features[i] in [feature]:

                #create a dictionary to store the average of each cluster
                average = {i: {t: [] for t in range(6)} for i in range(len(label.unique()))}
                #iterate through each sample
                for j in range(len(data)):
                    if np.sum(data[j,i,:]> 40)> 0 and features[i] == 'Hotels (except casino) and motels':
                        continue
                    if zip_name[j] in get_sample:
                        if label[j] in get_cluster:
                            if len(get_cluster) == len(label.unique()) and len(get_sample) == len(zip_name):
                                alpha = 1
                                s =150
                                style = '-'
                                lw = 3
                            else:
                                if zip_name[j] == get_sample[0]:
                                    alpha = 1
                                    s =150
                                    style = '-'
                                    lw = 3
                                else:
                                    alpha = 1
                                    s =150
                                    style = '-'
                                    lw = 3
                        else:
                            alpha = 1
                            s =150
                            style = '-'
                            lw = 3

                        c = [color[i] for i in cluster_label.iloc[j,:].tolist()]
                        m = [marker[i] for i in cluster_label.iloc[j,:].tolist()]

                        for t in range(6):
                            if donomiate is not None:
                                average[donomiate[j]][t].append(data[j,i,t])
                            if used_final_assignment:
                                average[label[j]][t].append(data[j,i,t])
                            else:
                                average[cluster_label.iloc[j,:].tolist()[t]][t].append(data[j,i,t])
                        
                        for length in np.arange(1,len(data[j,i,:])):
                            axs[count//3,count%3].plot([length -1, length],[data[j,i,length -1],data[j,i,length]],c = c[length-1],
                            alpha = alpha,lw = lw , ls = style)
                        
                        for t in range(data.shape[2]):
                            axs[count//3,count%3].scatter(t,data[j,i,t],ls = '--',c = c[t],alpha = alpha,s = s, marker = m[t])
                        axs[count//3,count%3].annotate(zip_name[j], (9.3,data[j,i,9] + 0.05), fontsize = 15)
                    else:
                        continue

                for cluster in range(len(label.unique())):
                    if cluster in get_cluster:
                        alpha = 0.2
                        agg_aplha = 1
                        agg_width = 3
                    else:
                        alpha = 0.3
                        agg_aplha = 0.3
                        agg_width = 1
        
                    tem = []
                    for t in range(6):
                        tem.append(np.array(agg(average[cluster][t])))

                    if get_sample == zip_name:
                        axs[count//3,count%3].plot(tem,c = color[cluster],ls = '--',lw=agg_width,
                                    alpha = agg_aplha)
                    
                    if cluster_center is not None:
                        if cluster in seleted_cluster:
                            axs[count//3,count%3].plot(cluster_center[cluster, i, :],
                                    c = color[cluster],ls = '--',lw=3,
                                    alpha = 1)
                            axs[count//3,count%3].scatter(0,cluster_center[cluster,i,0], s = 50, c = color[cluster], marker = marker[cluster], alpha = 0.3)
                            axs[count//3,count%3].scatter(range(data.shape[2])[-1],cluster_center[cluster,i,-1], s = 50, c = color[cluster], marker = marker[cluster], alpha = 0.3)
                if count == 1:
                    axs[count//3,count%3].set_title('SLA in 3-cluster setting \n \n' +title[i])
                else:
                    axs[count//3,count%3].set_title(title[i])

                axs[count//3,count%3].set_xticks(np.arange(12), ['', '2008', '', '2010', '', '2012', '', '2014', '', '2016','',''])
                if count%3 == 0:
                    axs[count//3,count%3].set_ylabel('Feature Value', x = 0.5, y = 0.5)
                count += 1
    for ax in axs:
        for a in ax:
            a.grid(False)

    cluster_label = cluster_label2
    label = label2
    color = ['#0072b2','#cc79a7','#d55e00']
    marker = ['o','s','^']
    cluster_center = cluster_center2
    count = 0
    for feature in get_feature:
        for i in range(len(col)):
            if features[i] in [feature]:
                #create a dictionary to store the average of each cluster
                average = {i: {t: [] for t in range(6)} for i in range(len(label.unique()))}
                #iterate through each sample
                for j in range(len(data)):
                    if np.sum(data[j,i,:]> 40)> 0 and features[i] == 'Hotels (except casino) and motels':
                        continue
                    if zip_name[j] in get_sample:
                        if label[j] in get_cluster:
                            if len(get_cluster) == len(label.unique()) and len(get_sample) == len(zip_name):
                                alpha = 1
                                s =150
                                style = '-'
                                lw = 3
                            else:
                                if zip_name[j] == get_sample[0]:
                                    alpha = 1
                                    s =150
                                    style = '-'
                                    lw = 3
                                else:
                                    alpha = 1
                                    s =150
                                    style = '-'
                                    lw = 3
                        else:
                            alpha = 1
                            s =150
                            style = '-'
                            lw = 3

                        c = [color[i] for i in cluster_label.iloc[j,:].tolist()]
                        m = [marker[i] for i in cluster_label.iloc[j,:].tolist()]

                        for t in range(6):
                            if donomiate is not None:
                                average[donomiate[j]][t].append(data[j,i,t])
                            if used_final_assignment:
                                average[label[j]][t].append(data[j,i,t])
                            else:
                                average[cluster_label.iloc[j,:].tolist()[t]][t].append(data[j,i,t])
                        for length in np.arange(1,len(data[j,i,:])):
                            axs[count//3+2,count%3].plot([length -1, length],[data[j,i,length -1],
                                                                   data[j,i,length]],
                                                                   c = c[length-1],alpha = alpha,lw = lw , ls = style)
                        
                        for t in range(data.shape[2]):
                            axs[count//3+2,count%3].scatter(t,data[j,i,t],ls = '--',c = c[t],alpha = alpha,s = s, marker = m[t])
                        axs[count//3+2,count%3].annotate(zip_name[j], (9.3,data[j,i,9] + 0.05), fontsize = 15)
                    else:
                        continue
                for cluster in range(len(label.unique())):
                    if cluster in get_cluster:
                        alpha = 0.2
                        agg_aplha = 1
                        agg_width = 3
                    else:
                        alpha = 0.3
                        agg_aplha = 0.3
                        agg_width = 1

                    tem = []
                    for t in range(6):
                        tem.append(np.array(agg(average[cluster][t])))

                    if get_sample == zip_name:
                        axs[count//3+2,count%3].plot(tem,c = color[cluster],ls = '--',lw=agg_width,
                                    alpha = agg_aplha)
                    
                    if cluster_center2 is not None:
                        if cluster in seleted_cluster:
                            axs[count//3+2,count%3].plot(cluster_center2[cluster, i, :],
                                    c = color[cluster],ls = '--',lw=3,
                                    alpha = 1)
                            axs[count//3+2,count%3].scatter(0,cluster_center2[cluster,i,0], s = 50, c = color[cluster], marker = marker[cluster], alpha = 0.3)
                            axs[count//3+2,count%3].scatter(range(data.shape[2])[-1],cluster_center2[cluster,i,-1], s = 50, c = color[cluster], marker = marker[cluster], alpha = 0.3)
                axs[count//3+2,count%3].set_title(title[i])

                axs[count//3+2,count%3].set_xticks(np.arange(12), ['', '2008', '', '2010', '', '2012', '', '2014', '', '2016','',''])
                legend_elements = [
                Line2D([0], [0], marker="s", color='#cc79a7', ls = '--',
                        label='(Stable) Hotels and Restaurants',
                        markerfacecolor='#cc79a7'),
                Line2D([0], [0], marker="^", color='#d55e00', ls = '--', 
                        label='(Increasing) Daliy Needs',
                            markerfacecolor='#d55e00'),
                Line2D([0], [0], marker="o", color='#0072b2', ls = '--', 
                        label='Low (and Declining) Business Activity',
                            markerfacecolor='#0072b2'),
                    ]
                if count%3 == 0:
                    axs[count//3+2,count%3].set_ylabel(f'Feature Value')
                if count >= 3:
                    axs[count//3+2,count%3].set_xlabel('Year')
                count += 1  
    #reduce margin
    fig.subplots_adjust(wspace=0.1, hspace=0.6)
    axs[2,1].set_title('\n \n  \n DC (L = 1) in 3-cluster setting \n \n ' + title[1], fontsize = 20)
    axs[2,2].legend(handles=legend_elements,fontsize =18, bbox_to_anchor=(1, 4.9))
    return fig

    # helper function to plot the geographical distribution of the cluster
def plot_sptial_distribution(data, cluster, cluster_label, cluster_name):
    plt.figure(figsize = (10,20))
    data[data['zip'].isin(cluster_label[[False if cluster in i[:-1].tolist() else True for row, 
                                        i in cluster_label.iterrows()]].index.astype(str))].plot(ax = plt.gca(), color = 'grey')
    data[~data['zip'].isin(cluster_label.index.astype(str))].plot(ax = plt.gca(), color = 'grey')
    for i, row in cluster_label[[True if cluster in i[:-1].tolist() else False for row, i in cluster_label.iterrows()]].iterrows():
        number_of_year = (row[:-1] == cluster).sum()
        if str(i) in data['zip'].values:
            data[data['zip']==str(i)].plot(ax = plt.gca(), color = 'red',
                                                alpha = number_of_year/10)
            #annotate as the centroid of the polygon
            x = data[data['zip']==str(i)].geometry.centroid.x.values[0]
            y = data[data['zip']==str(i)].geometry.centroid.y.values[0]
            plt.text(x,y,number_of_year , fontsize = 20)


    data.geometry.boundary.plot(ax = plt.gca(), color = 'white')
    plt.title(f'Geographical distribution of ZIP in \n {cluster_name[cluster]}',fontsize = 30)  
    from matplotlib.patches import Patch
    custom_lines = [Patch(facecolor='red', edgecolor='black',label='Cluster group'),Patch(facecolor='grey', edgecolor='black',label='Other groups')]

    plt.legend(custom_lines, [cluster_name[cluster][1:-1], 'Other groups'],fontsize = 25, loc = 'lower left')
    plt.axis('off')
    return plt

def waterfall_plot_center(data_array, sample = '0', generate = 100, cluster_label = None):
    features = ['Full-service restaurants','Supermarkets and grocery stores',
                'Religious organizations','Child day care services','Hotels (except casino) and motels',
                'Beer, wine, and liquor stores']
    tem = 0
    if sample is not None:
        zip_code = [1,2,3]
        zip_code = [str(i) for i in zip_code]
        # index of sample in zip_code
        index = zip_code.index(sample)
        data = data_array[index,:,:].T
    #seven color
    colors = ['b','r','g','m','orange','y','pink']
    df = pd.DataFrame(data, columns=features)
    output = df.iloc[:,0]
    #for each column in df
    for j in range(1,df.shape[1]):
        pre = df.iloc[:,j-1]
        cur = df.iloc[:,j]
        #ten new list linear interpolate between pre and cur
        new_list = []

        for i in range(generate):
            new_list.append(pre + (cur-pre)/generate*i)

        #create a df with column as column name + i
        new_df = pd.DataFrame(new_list).T
        new_df.columns = [features[j]+'_'+str(i) for i in range(generate)]
        #concatenate before current col
        output = pd.concat([output,new_df,df.iloc[:,j]],axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for z in range(0 ,output.shape[1]):
        x = [2007,2008,2009,2010,2011,2012,2013,2014,2015,2016]
        y = output.iloc[:,z]
        #if output.columns[z] does not have number
        if not any(char.isdigit() for char in output.columns[z]):
            color = ['yellow','green','blue','red','purple','orange','brown','pink','olive','cyan','yellow','grey','black','white','grey','black','white']
            ax.plot(x, y, zs=z, zdir='z',color = 'black',alpha=0.8,lw = 3)
            ax.scatter(x, y, zs=z, zdir='z',color = 'black',alpha=1)
            color = [color[int(sample)-1] for i in cluster_label.tolist()]
            tem += 1

        else:
            ax.plot(x, y, zs=z, zdir='z',color='grey',alpha=0.3)
    #change z axis label at 0, 20,40,60,80,100,120 to features
    ax.set_zticks([0*generate,generate,2*generate,3*generate,4*generate,5*generate], minor=False)
    ax.set_zticklabels(features, fontdict=None, minor=False, rotation=-90)
    ax.view_init(azim= 45, vertical_axis='y')
    #title
    labels = ['\'Low and Decline Business Activity',
                '\'Stable Hotels and Restaurants',
                '\'Increasing Daily Needs']

    ax.set_title('Dynamic Cluster Center of \n' + labels[int(sample)-1] + '\'', fontsize = 25)
    ax.set_xlabel('Year',fontsize =35, labelpad = 35)
    ax.set_xticks([2007,2008,2009,2010,2011,2012,2013,2014,2015,2016], minor=False)
    ax.set_xticklabels(['2007','2008','','2010','','2012','','2014','','2016'], fontdict=None, minor=False, rotation=-90)
    ax.set_ylabel('Normalized Value', labelpad = -30, fontsize = 35)
    ax.set_yticks(np.arange(0,1,0.2))
    #increase label size
    ax.tick_params(axis='z', which='major', labelsize=25)
    ax.tick_params(axis='x', which='major', labelsize=25)
    ax.tick_params(axis='y', which='major', labelsize=25)
    #increase title size
    ax.title.set_size(40)
    #increase margin
    plt.subplots_adjust(left=0.4, right=0.9, top=0.9, bottom=0.1)
    legend_elements = [Line2D([0], [0], color='black', marker='o', label='Cluster Center')]
    ax.legend(handles=legend_elements, loc='upper left',fontsize = 25)
    plt.tight_layout()
    return plt
