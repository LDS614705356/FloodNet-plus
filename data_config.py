
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = ""
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'FloodNet':
            # self.label_transform = "norm"
            self.root_dir = '/home/ljk/VQA/GoogleDrive'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='FloodNet')
    print(data.data_name)
    print(data.root_dir)
    # print(data.label_transform)