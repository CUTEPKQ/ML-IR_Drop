import os, re, bisect, gzip
import numpy as np
import pandas as pd
from src.util import save
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

class Paraser:
    def __init__(self, root_dir, arg, save_name):
        self.root_dir = root_dir
        self.save_name = save_name
        self.save_path = arg.save_path
        self.plot = arg.plot
        self.final_test = arg.final_test
        
        self.coordinate_x = []
        self.coordinate_y = []
        self.irdrop_map = None
  
        self.total_power_map = None
        # add new feature
        # leakage_power  switching_power  internal_power
        self.leakage_powe_map = None
        self.switching_power_map = None
        self.internal_power_map = None
        #////
        self.eff_res_VDD_map = None
        self.eff_res_VSS_map = None
        self.instance_count = None
        self.instance_IR_drop = None

    def get_IR_drop_features(self):
        max_size = (1000, 1000)
        self.irdrop_map = np.zeros(max_size)

        self.instance_count = np.zeros(max_size)
        self.instance_IR_drop = np.empty(max_size, dtype=object)
        for i in np.ndindex(max_size):
            self.instance_IR_drop[i] = []
        self.instance_name = np.empty(max_size, dtype=object)
        for i in np.ndindex(max_size):
            self.instance_name[i] = []
        self.total_power_map = np.zeros(max_size)

        self.leakage_powe_map= np.zeros(max_size)
        self.switching_power_map = np.zeros(max_size) 
        self.internal_power_map = np.zeros(max_size)

        self.eff_res_VDD_map = np.zeros(max_size)
        self.eff_res_VSS_map = np.zeros(max_size)

        self.coordinate_x = np.arange(0,max_size[0],1.44)
        self.coordinate_y = np.arange(0,max_size[1],1.152)      # based on the row height from LEF

        # try:
        #     if 'nvdla' in self.root_dir:
        #         data_power = pd.read_csv(os.path.join(self.root_dir, 'NV_nvdla.inst.power.rpt.gz'),sep='\s+',header=1, compression='gzip')
        #     elif 'Vortex' in self.root_dir:
        #         data_power = pd.read_csv(os.path.join(self.root_dir, 'Vortex.inst.power.rpt.gz'),sep='\s+',header=1, compression='gzip')
        #     else:
        #         data_power = pd.read_csv(os.path.join(self.root_dir, 'pulpino_top.inst.power.rpt.gz'),sep='\s+',header=1, compression='gzip')
        #     data_r = pd.read_csv(os.path.join(self.root_dir, 'eff_res.rpt.gz'),sep='\s+', low_memory=False, compression='gzip')
        #     if not self.final_test:
        #         data_ir = pd.read_csv(os.path.join(self.root_dir, 'static_ir.gz'),sep='\s+', compression='gzip')
        # except Exception as e:
        #     print('one of the report not exists')
        #     return 0    

        try:
            if 'nvdla' in self.root_dir:
                data_power = pd.read_csv(os.path.join(self.root_dir, 'NV_nvdla.inst.power.rpt'),sep='\s+',header=1)
            elif 'Vortex' in self.root_dir:
                data_power = pd.read_csv(os.path.join(self.root_dir, 'Vortex.inst.power.rpt'),sep='\s+',header=1)
            else:
                data_power = pd.read_csv(os.path.join(self.root_dir, 'pulpino_top.inst.power.rpt'),sep='\s+',header=1)
            data_r = pd.read_csv(os.path.join(self.root_dir, 'eff_res.rpt'),sep='\s+', low_memory=False)
            if not self.final_test:
                data_ir = pd.read_csv(os.path.join(self.root_dir, 'static_ir'),sep='\s+')
        except Exception as e:
            print('one of the report not exists')
            return 0    

        
        # try:
        #     if 'nvdla' in self.root_dir:
        #         data_power = pd.read_csv(os.path.join(self.root_dir, 'NV_nvdla.inst.power.rpt'),sep='\s+',header=1)
        #     else:
        #         # '\s+' 表示使用空格或多个空格作为分隔符, header=1 指定了CSV文件的第一行（索引从0开始）作为列名
        #         data_power = pd.read_csv(os.path.join(self.root_dir, 'pulpino_top.inst.power.rpt'),sep='\s+',header=1)
            
        #     # low_memory=False会加快读取速度，占用更多内存; low_memory=True会降低读取速度，占用更少内存
        #     data_r = pd.read_csv(os.path.join(self.root_dir, 'eff_res.rpt'),sep='\s+', low_memory=False) 
        #     data_ir = pd.read_csv(os.path.join(self.root_dir, 'static_ir'),sep='\s+')


        max_x = 0
        max_y = 0
        # parse inst.power.rpt        leakage_power  switching_power  internal_power
        total_power = data_power['total_power']
        leakage_power = data_power['leakage_power']
        switching_power = data_power['switching_power']
        internal_power = data_power['internal_power']


        bbox = data_power['bbox']
        name = data_power['*inst_name']
        if self.final_test:
            for i,j1,j2,j3,j4,k in zip(bbox, total_power,leakage_power,switching_power,internal_power, name):
                x1, y1, x2, y2 = i[1:-1].split(',')
                x = (float(x1)+float(x2))/2
                y = (float(y1)+float(y2))/2
                gcell_x = bisect.bisect_left(self.coordinate_x, float(x)-10)
                gcell_y = bisect.bisect_left(self.coordinate_y, float(y)-10)
                if gcell_x > max_x:
                    max_x = gcell_x
                if gcell_y > max_y: 
                    max_y = gcell_y
                self.total_power_map[gcell_x, gcell_y] += j1
                self.leakage_powe_map[gcell_x, gcell_y] +=j2
                self.switching_power_map[gcell_x, gcell_y] +=j3
                self.internal_power_map[gcell_x, gcell_y] +=j4

                self.instance_name[gcell_x, gcell_y].append(k)
                self.instance_count[gcell_x, gcell_y] += 1
            self.total_power_map = self.total_power_map[0:max_x+1,0:max_y+1]
            self.leakage_powe_map = self.leakage_powe_map[0:max_x+1,0:max_y+1]
            self.switching_power_map = self.switching_power_map[0:max_x+1,0:max_y+1]
            self.internal_power_map = self.internal_power_map[0:max_x+1,0:max_y+1]

            save(self.save_path, 'features/instance_count_from_power_rpt', self.save_name, self.instance_count)
            self.instance_name = np.concatenate(self.instance_name.ravel())
            # 以npz形式保存以节省空间
            instance_name_save_path = os.path.join(self.save_path, 'features/instance_name_from_power_rpt', self.save_name)
            if not os.path.exists(os.path.dirname(instance_name_save_path)):
                os.makedirs(os.path.dirname(instance_name_save_path))
            np.savez_compressed(instance_name_save_path, instance_name=self.instance_name)
        else:
            for i,j1,j2,j3,j4 in zip(bbox, total_power,leakage_power,switching_power,internal_power):
                x1, y1, x2, y2 = i[1:-1].split(',')
                x = (float(x1)+float(x2))/2
                y = (float(y1)+float(y2))/2
                gcell_x = bisect.bisect_left(self.coordinate_x, float(x)-10)
                gcell_y = bisect.bisect_left(self.coordinate_y, float(y)-10)
                if gcell_x > max_x:
                    max_x = gcell_x
                if gcell_y > max_y: 
                    max_y = gcell_y
                self.total_power_map[gcell_x, gcell_y] += j1
                self.leakage_powe_map[gcell_x, gcell_y] +=j2
                self.switching_power_map[gcell_x, gcell_y] +=j3
                self.internal_power_map[gcell_x, gcell_y] +=j4

            self.total_power_map = self.total_power_map[0:max_x+1,0:max_y+1]
            self.leakage_powe_map = self.leakage_powe_map[0:max_x+1,0:max_y+1]
            self.switching_power_map = self.switching_power_map[0:max_x+1,0:max_y+1]
            self.internal_power_map = self.internal_power_map[0:max_x+1,0:max_y+1]
          
        save(self.save_path, 'features/total_power', self.save_name, self.total_power_map)
        save(self.save_path, 'features/leakage_power', self.save_name, self.leakage_powe_map)
        save(self.save_path, 'features/switching_power', self.save_name, self.switching_power_map)
        save(self.save_path, 'features/internal_power', self.save_name, self.internal_power_map)



        # 要预测的是vdd_drop和gnd_bounce。
        # parse static_ir 因为report开头的井号被当作单独一列，这里的key也往前移了一位。
        # 即data_ir['inst_vdd']对应的数据是vdd_drop，而不是inst_vdd。
        # 类似地，data_ir['pwr_net']对应的数据是location，而不是pwr_net，data_ir['location']对应的数据是instance name，而不是location。
        
        if not self.final_test: # 最终测试的时候没有static_ir这个文件。
            vdd_drop = data_ir['inst_vdd']     
            gnd_bounce = data_ir['vdd_drop']     
            location = data_ir['pwr_net'] 
            name = data_ir['location']

     
            for i,j,k,l in zip(location, vdd_drop,gnd_bounce, name):
                x, y = i.split(',')
                gcell_x = bisect.bisect_left(self.coordinate_x, float(x)-10) # -10是因为版图周围有一圈10um的padding。
                gcell_y = bisect.bisect_left(self.coordinate_y, float(y)-10)                                                            
                if j+k > self.irdrop_map[gcell_x, gcell_y]:
                     self.irdrop_map[gcell_x, gcell_y] = j+k
                self.instance_IR_drop[gcell_x, gcell_y].append(j+k)
                self.instance_name[gcell_x, gcell_y].append(l)
                self.instance_count[gcell_x, gcell_y] += 1

            self.irdrop_map = self.irdrop_map[0:max_x+1,0:max_y+1]
            
            save(self.save_path, 'features/irdrop', self.save_name, self.irdrop_map)
            self.instance_IR_drop = np.concatenate(self.instance_IR_drop.ravel())
            
            
            save(self.save_path, 'features/instance_IR_drop', self.save_name, self.instance_IR_drop)

            self.instance_count = self.instance_count[0:max_x+1,0:max_y+1]
            save(self.save_path, 'features/instance_count', self.save_name, self.instance_count)
            self.instance_name = np.concatenate(self.instance_name.ravel())
            # 以npz形式保存以节省空间
            instance_name_save_path = os.path.join(self.save_path, 'features/instance_name', self.save_name)
            if not os.path.exists(os.path.dirname(instance_name_save_path)):
                os.makedirs(os.path.dirname(instance_name_save_path))
            np.savez_compressed(instance_name_save_path, instance_name=self.instance_name)

        # parse eff_res.rpt
        vdd_r = data_r['loop_r']
        vss_r = data_r['vdd_r']
        location_x = data_r['gnd_r']
        location_y = data_r['vdd(x']

        for i,j,k,l in zip(location_x, location_y, vdd_r, vss_r):
            x = i[1:]
            y = j
            if i == '-' or j == '-' or k == '-' or l == '-':
                continue
            gcell_x = bisect.bisect_left(self.coordinate_x, float(x)-10)
            gcell_y = bisect.bisect_left(self.coordinate_y, float(y)-10)
            self.eff_res_VDD_map[gcell_x, gcell_y] += float(k)
            self.eff_res_VSS_map[gcell_x, gcell_y] += float(l)

        self.eff_res_VDD_map = self.eff_res_VDD_map[0:max_x+1,0:max_y+1]
        self.eff_res_VSS_map = self.eff_res_VSS_map[0:max_x+1,0:max_y+1]

        save(self.save_path, 'features/eff_res_VDD', self.save_name, self.eff_res_VDD_map)
        save(self.save_path, 'features/eff_res_VSS', self.save_name, self.eff_res_VSS_map)

        # # for visualization
        # if self.plot:
        #     if not os.path.exists(os.path.join(self.save_path, 'visual', self.save_name)):
        #         os.makedirs(os.path.join(self.save_path, 'visual', self.save_name))
        #     fig = sns.heatmap(data=self.eff_res_VDD_map, cmap="rainbow").get_figure()
        #     fig.savefig(os.path.join(self.save_path,'visual', self.save_name,'eff_res_VDD.png'), dpi=100)
        #     plt.close()

        #     fig = sns.heatmap(data=self.eff_res_VSS_map, cmap="rainbow").get_figure()
        #     fig.savefig(os.path.join(self.save_path,'visual', self.save_name,'eff_res_VSS.png'), dpi=100)
        #     plt.close()

        #     fig = sns.heatmap(data=self.total_power_map, cmap="rainbow").get_figure()
        #     fig.savefig(os.path.join(self.save_path,'visual', self.save_name,'total_power.png'), dpi=100)
        #     plt.close()

        #     fig = sns.heatmap(data=self.VDD_drop_map, cmap="rainbow").get_figure()
        #     fig.savefig(os.path.join(self.save_path,'visual', self.save_name,'VDD_drop.png'), dpi=100)
        #     plt.close()

        #     fig = sns.heatmap(data=self.GND_bounce_map, cmap="rainbow").get_figure()
        #     fig.savefig(os.path.join(self.save_path,'visual', self.save_name,'GND_bounce.png'), dpi=100)
        #     plt.close()
