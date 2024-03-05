from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

from configs import config_3d
args = config_3d.args
vector_bins = args.vector_bins


class up_conv_3d(nn.Module):
	"""
	Up Convolution Block
	"""
	def __init__(self, in_ch, out_ch):
		super(up_conv_3d, self).__init__()
		self.up = nn.Sequential(
			nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=True),
			nn.BatchNorm3d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.up(x)
		return x

class down_conv_3d(nn.Module):
	"""
	Up Convolution Block
	"""
	def __init__(self, in_ch, out_ch):
		super(down_conv_3d, self).__init__()
		self.down = nn.Sequential(
			nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=True),
			nn.BatchNorm3d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.down(x)
		return x

class res_conv_block_3d(nn.Module):
	"""
	Res Convolution Block
	"""
	def __init__(self, in_ch, out_ch):
		super(res_conv_block_3d, self).__init__()
		
		self.conv1 = nn.Sequential(
			nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
			nn.BatchNorm3d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm3d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv3d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
			nn.BatchNorm3d(out_ch))
		
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out += residual
		out = self.relu(out)
		return out

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size) 
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # print(x.shape, hidden[0].shape, hidden[1].shape)
        batch_size, seq_len, _ = x.shape
        out, hidden = self.lstm(x, hidden)
        # print(out.shape)
        if seq_len == 1:
            out1 = self.fc1(out[:, 0, :])
            out2 = self.fc1(out[:, 0, :])
            out3 = self.fc1(out[:, 0, :])
        else:
            out1 = self.fc1(out[:, 0, :])
            out2 = self.fc2(out[:, 1, :])
            out3 = self.fc3(out[:, 2, :])
        # print(out.shape)
        return [out1, out2, out3], hidden

class CSFL_Net_3D(nn.Module):
	"""
	Res_U_Net - Basic Implementation
	Paper : https://arxiv.org/abs/1505.04597
	"""

	def __init__(self, in_ch=1, out_ch=1, freeze_net = False):
		super(CSFL_Net_3D, self).__init__()
  
		self.n1 = 32
		filters = [self.n1, self.n1 * 2, self.n1 * 4, self.n1 * 8, self.n1 * 16]
		
		# RNN parameter
		self.layer_num_RNN = 1
		self.seq_len = 3

		self.image_hidden_size = 1*4*4
		self.input_size_RNN = filters[0]*self.image_hidden_size
		self.hidden_size_RNN = 100
		self.output_size_direction = vector_bins
		self.output_size_radius = 1



		self.conv_input = nn.Sequential(nn.Conv3d(in_ch, filters[0], kernel_size=3, stride=1, padding=1, bias=True), 
										nn.BatchNorm3d(filters[0]),
										nn.ReLU(inplace=True))

		self.Conv1 = res_conv_block_3d(filters[0], filters[0])
		self.Down1 = down_conv_3d(filters[0], filters[1])

		self.Conv2 = res_conv_block_3d(filters[1], filters[1])
		self.Down2 = down_conv_3d(filters[1], filters[2])

		self.Conv3 = res_conv_block_3d(filters[2], filters[2])
		self.Down3 = down_conv_3d(filters[2], filters[3])

		self.Conv4 = res_conv_block_3d(filters[3], filters[3])
		self.Down4 = down_conv_3d(filters[3], filters[4])

		self.Conv5_1 = res_conv_block_3d(filters[4], filters[4])
		self.Conv5_2 = res_conv_block_3d(filters[4], filters[4])
		self.Conv5_3 = res_conv_block_3d(filters[4], filters[4])

		self.Up5 = up_conv_3d(filters[4], filters[3])
		self.Up_conv5 = res_conv_block_3d(filters[3], filters[3])

		self.Up4 = up_conv_3d(filters[3], filters[2])
		self.Up_conv4 = res_conv_block_3d(filters[2], filters[2])

		self.Up3 = up_conv_3d(filters[2], filters[1])
		self.Up_conv3 = res_conv_block_3d(filters[1], filters[1])

		self.Up2 = up_conv_3d(filters[1], filters[0])
		self.Up_conv2 = res_conv_block_3d(filters[0], filters[0])
		
		# seg block
		self.Conv6_1 = nn.Sequential(nn.Conv3d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
										nn.BatchNorm3d(filters[0]),
										nn.ReLU(inplace=True))
		self.Conv6_2 = nn.Sequential(nn.Conv3d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
										nn.BatchNorm3d(filters[0]),
										nn.ReLU(inplace=True))
		self.Conv6_out = nn.Sequential(nn.Conv3d(filters[0], out_ch, kernel_size=3, stride=1, padding=1),
										nn.Sigmoid())
		# self.Conv6_out = nn.Sequential(nn.Conv3d(filters[0], out_ch, kernel_size=3, stride=1, padding=1))

		# centerline block
		self.Conv7_1 = nn.Sequential(nn.Conv3d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
										nn.BatchNorm3d(filters[0]),
										nn.ReLU(inplace=True))
		self.Conv7_2 = nn.Sequential(nn.Conv3d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
										nn.BatchNorm3d(filters[0]),
										nn.ReLU(inplace=True))
		self.Conv7_out = nn.Sequential(nn.Conv3d(filters[0], out_ch, kernel_size=3, stride=1, padding=1),
										nn.Sigmoid())
		# self.Conv7_out = nn.Sequential(nn.Conv3d(filters[0], out_ch, kernel_size=3, stride=1, padding=1))
		if freeze_net:
			print("freezing")
			freeze(self)			

		
		# direction head
		self.Tracer1 = nn.Sequential(nn.Conv3d(filters[4], filters[0], kernel_size=3, stride=1,  padding=1, bias=True), 
										nn.BatchNorm3d(filters[0]),
										nn.ReLU(inplace=True))
		self.Tracer2 = nn.Sequential(nn.Conv3d(filters[0], filters[0], kernel_size=3, stride=1,  padding=1, bias=True), 
										nn.BatchNorm3d(filters[0]),
										nn.ReLU(inplace=True))
		self.Tracer3 = nn.Sequential(nn.Conv3d(filters[0], filters[0], kernel_size=3, stride=1,  padding=1, bias=True),
										nn.BatchNorm3d(filters[0]),
										nn.ReLU(inplace=True))
		self.Tracer4_1 = nn.Linear(filters[0]*self.image_hidden_size, self.output_size_direction)
		self.Tracer4_2 = nn.Linear(filters[0]*self.image_hidden_size, self.output_size_direction)	
		self.Tracer4_3 = nn.Linear(filters[0]*self.image_hidden_size, self.output_size_direction)

		self.Tracer4_RNN_1 = LSTM(self.input_size_RNN, self.hidden_size_RNN, self.output_size_direction, self.seq_len)
		self.Tracer4_RNN_2 = LSTM(self.input_size_RNN, self.hidden_size_RNN, self.output_size_direction, self.seq_len)
		self.Tracer4_RNN_3 = LSTM(self.input_size_RNN, self.hidden_size_RNN, self.output_size_direction, self.seq_len)

		# radius head
		self.Radius1 = nn.Sequential(nn.Conv3d(filters[4], filters[0], kernel_size=3, stride=1,  padding=1, bias=True), 
										nn.BatchNorm3d(filters[0]),
										nn.ReLU(inplace=True))
		self.Radius2 = nn.Sequential(nn.Conv3d(filters[0], filters[0], kernel_size=3, stride=1,  padding=1, bias=True), 
										nn.BatchNorm3d(filters[0]),
										nn.ReLU(inplace=True))
		self.Radius3 = nn.Sequential(nn.Conv3d(filters[0], filters[0], kernel_size=3, stride=1,  padding=1, bias=True),
										nn.BatchNorm3d(filters[0]),
										nn.ReLU(inplace=True))
		self.Radius4 = nn.Sequential(nn.Linear(filters[0]*self.image_hidden_size, 1, bias = True))

		self.Radius4_RNN = LSTM(self.input_size_RNN, self.hidden_size_RNN, self.output_size_radius, self.seq_len)
		# RNN Block


	def forward(self, x, mode = 'train'):
		outputs_img = []
		outputs_direction_radius_single = []
		hidden_direction_radius = []
		outputs_direction_radius_multi = []
		batch_size, seq_len, c, d, h, w = x.size()
		
		if mode == 'train':
			for t in range(seq_len): 
				input_image = x[:, t, :, :, :, :]

				d0 = self.conv_input(input_image)

				e1 = self.Conv1(d0)
				e1_d = self.Down1(e1)

				e2 = self.Conv2(e1_d)
				e2_d = self.Down2(e2)

				e3 = self.Conv3(e2_d)
				e3_d = self.Down3(e3)

				e4 = self.Conv4(e3_d)
				e4_d = self.Down4(e4)

				e5_2 = self.Conv5_1(e4_d)
				e5_1 = self.Conv5_2(e5_2)
				e5 = self.Conv5_3(e5_1)
				# e5 = self.Conv5_1(e4_d)

				# print(e5.shape)

				d5 = self.Up5(e5)
				d5 = torch.add(e4, d5)
				d5 = self.Up_conv5(d5)

				d4 = self.Up4(d5)
				d4 = torch.add(e3, d4)
				d4 = self.Up_conv4(d4)

				d3 = self.Up3(d4)
				d3 = torch.add(e2, d3)
				d3 = self.Up_conv3(d3)

				d2 = self.Up2(d3)
				d2 = torch.add(e1, d2)
				d2 = self.Up_conv2(d2)


				# exist block
				d6_1 = self.Conv6_1(d2)	
				d6_2 = self.Conv6_2(d6_1)	
				d_seg = self.Conv6_out(d6_2)

				# centerline block
				d7_1 = self.Conv7_1(d2)	
				d7_2 = self.Conv7_2(d7_1)	
				d_centerline = self.Conv7_out(d7_2)	

				# direction block
				t1 = self.Tracer1(e5)
				t2 = self.Tracer2(t1)
				t3 = self.Tracer3(t2)
				t3_flatten = t3.view(-1, self.n1*self.image_hidden_size)
				t_angle_1 = self.Tracer4_1(t3_flatten)
				t_angle_2 = self.Tracer4_2(t3_flatten)
				t_angle_3 = self.Tracer4_3(t3_flatten)

				# radius block
				r1 = self.Radius1(e5)
				r2 = self.Radius2(r1)
				r3 = self.Radius3(r2)
				r3_flatten = r3.view(-1, self.n1*self.image_hidden_size)
				r_radius = self.Radius4(r3_flatten)


				outputs_img += [[d_seg, d_centerline]]
				outputs_direction_radius_single += [[t_angle_1, t_angle_2, t_angle_3, r_radius]]
				hidden_direction_radius += [[t3_flatten, r3_flatten]]


			# 输出 image output
			output_seg = outputs_img[0][0].unsqueeze(1)
			output_centerline = outputs_img[0][1].unsqueeze(1)
			for t in range(1, seq_len): 
				output_seg_temp = outputs_img[t][0].unsqueeze(1)
				output_centerline_temp = outputs_img[t][1].unsqueeze(1)

				output_seg = torch.cat([output_seg, output_seg_temp], dim=1)
				output_centerline = torch.cat([output_centerline, output_centerline_temp], dim=1)


			output_d1_single = outputs_direction_radius_single[0][0].unsqueeze(1)
			output_d2_single = outputs_direction_radius_single[0][1].unsqueeze(1)
			output_d3_single = outputs_direction_radius_single[0][2].unsqueeze(1)
			output_r_single = outputs_direction_radius_single[0][3].unsqueeze(1)
			for t in range(1, seq_len): 
				output_d1_single_temp = outputs_direction_radius_single[t][0].unsqueeze(1)
				output_d2_single_temp = outputs_direction_radius_single[t][1].unsqueeze(1)
				output_d3_single_temp = outputs_direction_radius_single[t][2].unsqueeze(1)
				output_r_single_temp = outputs_direction_radius_single[t][3].unsqueeze(1)

				output_d1_single = torch.cat([output_d1_single, output_d1_single_temp], dim=1)
				output_d2_single = torch.cat([output_d2_single, output_d2_single_temp], dim=1)
				output_d3_single = torch.cat([output_d3_single, output_d3_single_temp], dim=1)
				output_r_single = torch.cat([output_r_single, output_r_single_temp], dim=1)

			# 输出 direction/radius multi output
			input_t3 = hidden_direction_radius[0][0].unsqueeze(1)
			input_r3 = hidden_direction_radius[0][1].unsqueeze(1)
			for t in range(1, seq_len): 
			
				input_t3_temp = hidden_direction_radius[t][0].unsqueeze(1)
				input_r3_temp = hidden_direction_radius[t][1].unsqueeze(1)

				input_t3 = torch.cat([input_t3, input_t3_temp], dim=1)
				input_r3 = torch.cat([input_r3, input_r3_temp], dim=1)

			# 计算RNN模块
			hidden_d1 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			cell_d1 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			hidden_d2 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			cell_d2 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			hidden_d3 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			cell_d3 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')

			hidden_r = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			cell_r = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')

			# hidden_d1 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to(device)
			# cell_d1 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to(device)
			# hidden_d2 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to(device)
			# cell_d2 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to(device)
			# hidden_d3 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to(device)
			# cell_d3 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to(device)

			# hidden_r = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to(device)
			# cell_r = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to(device)

			[output1_d1, output2_d1, output3_d1], (hidden_d1, cell_d1) = self.Tracer4_RNN_1(input_t3, (hidden_d1, cell_d1))
			[output1_d2, output2_d2, output3_d2], (hidden_d2, cell_d2) = self.Tracer4_RNN_2(input_t3, (hidden_d2, cell_d2))
			[output1_d3, output2_d3, output3_d3], (hidden_d3, cell_d3) = self.Tracer4_RNN_3(input_t3, (hidden_d3, cell_d3))

			[output1_r, output2_r, output3_r] , (hidden_r, cell_r) = self.Radius4_RNN(input_r3, (hidden_r, cell_r))


			output_d1 = torch.cat([output1_d1.unsqueeze(1), output2_d1.unsqueeze(1), output3_d1.unsqueeze(1)], dim=1)
			output_d2 = torch.cat([output1_d2.unsqueeze(1), output2_d2.unsqueeze(1), output3_d2.unsqueeze(1)], dim=1)
			output_d3 = torch.cat([output1_d3.unsqueeze(1), output2_d3.unsqueeze(1), output3_d3.unsqueeze(1)], dim=1)
			output_r = torch.cat([output1_r, output2_r, output3_r], dim=1)

			# print(output_seg.shape, output_centerline.shape, output_d1_single.shape,output_d2_single.shape,output_d3_single.shape, output_r_single.shape, output_d1.shape, output_d2.shape, output_d3.shape, output_r.shape)	

			return output_seg, output_centerline, output_d1_single, output_d2_single, output_d3_single, output_r_single, output_d1, output_d2, output_d3, output_r
		elif mode == 'test_dis':
			input_image = x[:, 0, :, :, :, :]

			d0 = self.conv_input(input_image)

			e1 = self.Conv1(d0)
			e1_d = self.Down1(e1)

			e2 = self.Conv2(e1_d)
			e2_d = self.Down2(e2)

			e3 = self.Conv3(e2_d)
			e3_d = self.Down3(e3)

			e4 = self.Conv4(e3_d)
			e4_d = self.Down4(e4)

			e5_2 = self.Conv5_1(e4_d)
			e5_1 = self.Conv5_2(e5_2)
			e5 = self.Conv5_3(e5_1)
				

			d5 = self.Up5(e5)
			d5 = torch.add(e4, d5)
			d5 = self.Up_conv5(d5)

			d4 = self.Up4(d5)
			d4 = torch.add(e3, d4)
			d4 = self.Up_conv4(d4)

			d3 = self.Up3(d4)
			d3 = torch.add(e2, d3)
			d3 = self.Up_conv3(d3)

			d2 = self.Up2(d3)
			d2 = torch.add(e1, d2)
			d2 = self.Up_conv2(d2)


			# exist block
			d6_1 = self.Conv6_1(d2)	
			d6_2 = self.Conv6_2(d6_1)	
			d_seg = self.Conv6_out(d6_2)

			# centerline block
			d7_1 = self.Conv7_1(d2)	
			d7_2 = self.Conv7_2(d7_1)	
			d_centerline = self.Conv7_out(d7_2)	
			return d_seg, d_centerline
		
		elif mode == 'test_d':
			for t in range(seq_len): 
				input_image = x[:, t, :, :, :, :]

				d0 = self.conv_input(input_image)

				e1 = self.Conv1(d0)
				e1_d = self.Down1(e1)

				e2 = self.Conv2(e1_d)
				e2_d = self.Down2(e2)

				e3 = self.Conv3(e2_d)
				e3_d = self.Down3(e3)

				e4 = self.Conv4(e3_d)
				e4_d = self.Down4(e4)

				e5_2 = self.Conv5_1(e4_d)
				e5_1 = self.Conv5_2(e5_2)
				e5 = self.Conv5_3(e5_1)

				# direction block
				t1 = self.Tracer1(e5)
				t2 = self.Tracer2(t1)
				t3 = self.Tracer3(t2)
				t3_flatten = t3.view(-1, self.n1*self.image_hidden_size)

				# radius block
				r1 = self.Radius1(e5)
				r2 = self.Radius2(r1)
				r3 = self.Radius3(r2)
				r3_flatten = r3.view(-1, self.n1*self.image_hidden_size)

				hidden_direction_radius += [[t3_flatten, r3_flatten]]
			
			# 输出 direction/radius multi output
			input_t3 = hidden_direction_radius[0][0].unsqueeze(1)
			input_r3 = hidden_direction_radius[0][1].unsqueeze(1)
			for t in range(1, seq_len): 
			
				input_t3_temp = hidden_direction_radius[t][0].unsqueeze(1)
				input_r3_temp = hidden_direction_radius[t][1].unsqueeze(1)

				input_t3 = torch.cat([input_t3, input_t3_temp], dim=1)
				input_r3 = torch.cat([input_r3, input_r3_temp], dim=1)

			# 计算RNN模块
			hidden_d1 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			cell_d1 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			hidden_d2 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			cell_d2 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			hidden_d3 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			cell_d3 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')

			hidden_r = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			cell_r = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')

   
			[output1_d1, output2_d1, output3_d1], (hidden_d1, cell_d1) = self.Tracer4_RNN_1(input_t3, (hidden_d1, cell_d1))
			[output1_d2, output2_d2, output3_d2], (hidden_d2, cell_d2) = self.Tracer4_RNN_2(input_t3, (hidden_d2, cell_d2))
			[output1_d3, output2_d3, output3_d3], (hidden_d3, cell_d3) = self.Tracer4_RNN_3(input_t3, (hidden_d3, cell_d3))

			[output1_r, output2_r, output3_r] , (hidden_r, cell_r) = self.Radius4_RNN(input_r3, (hidden_r, cell_r))


			output_d1 = torch.cat([output1_d1.unsqueeze(1), output2_d1.unsqueeze(1), output3_d1.unsqueeze(1)], dim=1)
			output_d2 = torch.cat([output1_d2.unsqueeze(1), output2_d2.unsqueeze(1), output3_d2.unsqueeze(1)], dim=1)
			output_d3 = torch.cat([output1_d3.unsqueeze(1), output2_d3.unsqueeze(1), output3_d3.unsqueeze(1)], dim=1)
			output_r = torch.cat([output1_r, output2_r, output3_r], dim=1)

			
			return output_d1, output_d2, output_d3, output_r

