# setup
from posixpath import basename
from shutil import which
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash import no_update, callback_context
import os
import glob
from plotly.subplots import make_subplots
import cv2
import base64
from io import BytesIO as _BytesIO
from PIL import Image
import numpy as np
from collections import defaultdict
from datetime import datetime

def pil2base64(im, enc_format='png', **kwargs):
	"""
	Converts a PIL Image into base64 string for HTML displaying
	:param im: PIL Image object
	:param enc_format: The image format for displaying. If saved the image will have that extension.
	:return: base64 encoding
	"""

	buff = _BytesIO()
	im.save(buff, format=enc_format, quality=92, **kwargs)
	encoded = base64.b64encode(buff.getvalue()).decode("utf-8")

	return encoded
		
def np2base64(np_array, enc_format='png', scalar=False, **kwargs):
	"""
	Converts a numpy image into base 64 string for HTML displaying
	:param np_array:
	:param enc_format: The image format for displaying. If saved the image will have that extension.
	:param scalar:
	:return:
	"""
	
	# Convert from 0-1 to 0-255
	if scalar:
		np_array = np.uint8(255 * np_array)
	else:
		np_array = np.uint8(np_array)

	im_pil = Image.fromarray(np_array)

	return pil2base64(im_pil, enc_format, **kwargs)

def convert_path_to_linux(path):
	path = path.replace("\\",'/')
	if 'AIR' in path:
		path = path.replace(path[:path.find('/AIR')],'')
	elif 'nas' in path:
		path = path.replace(path[:path.find('/nas')],'/media')
	elif 'V-Storage' in path:
		path = path.replace(path[:path.find('/V-Storage')],'/media/nfs')
	return path

class Viewer:
	MAX_FOLDERS = 4
	def __init__(self):
		self.image_directories = ['']*Viewer.MAX_FOLDERS
		self.current_images = [None]*Viewer.MAX_FOLDERS
		self.image_lists = [[]]*Viewer.MAX_FOLDERS
		self.image_lists_original = [[]]*Viewer.MAX_FOLDERS
		self.normalizer = [False]*Viewer.MAX_FOLDERS
		self.indexes = [0]*Viewer.MAX_FOLDERS
		self.skip_rate = 1
		self.total_folders = Viewer.MAX_FOLDERS
		self.current_scale = [1]*Viewer.MAX_FOLDERS
		self.slider_size = [4, 2, 1.5, 1.25, 1, 0.5, 0.25]
		self.current_filter = ''
		self.layout_type = 'Grid'
		self.grid_object = Tab1()
		self.total_paritions = [1]*Viewer.MAX_FOLDERS
		self.parition_number = [1]*Viewer.MAX_FOLDERS
	
	def reset_input(self):
		self.image_lists = self.image_lists_original.copy()
		self.indexes = [0]*Viewer.MAX_FOLDERS

	def setup(self):
		theme = dbc.themes.SOLAR
		css = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'

		# App Instance
		self.app = dash.Dash(name="viewer",
						external_stylesheets=[theme, css],
						suppress_callback_exceptions=True)
		self.create_layout()
		self.setup_callback()
		self.grid_object.setup_callback(self.app)
	
	def run(self):
			self.app.run_server(debug=True, host = "0.0.0.0", port=8080)
	

	def starting_layout(self):
		app = self.app
		navbar = self.get_navbar()
		folder_numbers = dbc.Card(
			[
				dbc.CardHeader("Total number of direcotries"),
				dbc.CardBody(
					[
						dbc.Select(
							id="total_folders",
							options=self.get_options(),
							value=self.total_folders,
						),
					]
				),
			],
			className="mt-3",
		)
		app.layout = dbc.Container(fluid=True, children=[
			html.H1("viewer", id="temp"),
			navbar,
			folder_numbers,
			html.Div(id="page-content", children=[])
			])
	
	def get_options(self):
		options = []
		for i in range(Viewer.MAX_FOLDERS):
			options.append({"label": f"{i+1}", "value" : i+1})
		return options

	def get_navbar(self):
		navbar = dbc.Nav(className="nav temp", children=[
			## about
			dbc.NavItem(html.Div([
				dbc.NavLink("About", href="/", id="about-popover", active=False),
				dbc.Popover(id="about", is_open=False, target="about-popover", children=[
					dbc.PopoverHeader("How it works"), dbc.PopoverBody("Allow you to visualize images from multiple folders")
				])
			])),
			# ## links
			# dbc.DropdownMenu(label="Links", nav=True, children=[
			#     dbc.DropdownMenuItem([html.I(className="fa fa-linkedin"), "  Contacts"], href=config.contacts, target="_blank"), 
			#     dbc.DropdownMenuItem([html.I(className="fa fa-github"), "  Code"], href=config.code, target="_blank"),
			#     dbc.DropdownMenuItem([html.I(className="fa fa-medium"), "  Tutorial"], href=config.tutorial, target="_blank")
			# ])
		])
		return navbar
	
	def get_input_layout(self):
		input_groups = []
		for i in range(self.total_folders):
			input_groups.append(
				dbc.InputGroup(
					[
						dbc.Input(
							value=f"{self.image_directories[i]}",
							size="sm",
							debounce=True,
							id=f"imagedir-input{i}",
						)
					]
				))
		input_layout = html.Div(
			input_groups
			)
		
		return input_layout
	
	def get_parition_options(self, index):
		if self.total_paritions[index]<=1:
			options = [{'label': '1', 'value' : 1}]
		else:
			options = []
			for val in range(self.total_paritions[index]):
				options.append({'label': f'{val+1}', 'value' : val+1})
		return options

	def parition_info(self, index):
		return html.Div(dbc.Select(
								id=f"Which Parition{index}",
								options=self.get_parition_options(index),
								value=1
							))

	
	def get_one_image_struct(self, index):
		one_image_struct = [
		dbc.Row(
					[
						html.Img(id=f'main-image{index}',
								src=self.resize_convert_bytearray(self.current_scale[index], index)),
					],),
		dbc.Row(
			dcc.Textarea(
						id=f'image_name{index}',
						value='',
						style={'width': '20%'}
					)
		),
		dbc.Row(
				[
				dbc.Col([
					dbc.Button(
					"<",
					color="primary",
					className="me-1",
					id=f"previous_image{index}",
					disabled=False,
				),
					dbc.Button(
						">",
						color="primary",
						className="me-1",
						id=f"next_image{index}",
						disabled=False,
					)]),
				dbc.Col(
					dbc.Input(type="number", min=1, max=max(len(self.image_lists[index]),1), id=f'numnav{index}', value=0, step=1, debounce=True), width=30
				),
				dbc.Col(
					html.P(id=f'index_counter{index}', children=f"{self.indexes[index]+1}/{len(self.image_lists[index])}"),
				),
				dbc.Col(
					dbc.Button(
								"Normalize",
								color="primary",
								className="me-1",
								id=f"Normalize Data{index}",
								disabled=False,
							)
				),
				dbc.Col([
						dbc.Input(type="int", id=f'Total Paritions{index}', debounce=True, value=self.total_paritions[index]),
						html.P(f"Total Paritions")
					]
				),
				dbc.Col([
					html.Div(id=f"Parition Number{index}", children=self.parition_info(index)),
					html.P(f"Parition Number")
				]
				),
				]),
		dbc.Spinner(
					self.get_slider(index)
		)
		]
		return one_image_struct


	def get_image_layout(self):       
		if self.layout_type=='Grid':
			return self.get_grid_layout()
		elif self.layout_type=='Horizontal':
			return self.get_horizontal_layout()
		elif self.layout_type=='Vertical':
			return self.get_vertical_layout()

	def get_grid_layout(self):
		image_layout =  html.Div(
							dbc.Container([
								dbc.Row(
									[
										dbc.Col(
											self.get_one_image_struct(0)
										),
										dbc.Col(
											self.get_one_image_struct(1)
										),
									]
								),
								dbc.Row(
									[
										dbc.Col(
											self.get_one_image_struct(2)
										),
										dbc.Col(
											self.get_one_image_struct(3)
										),
									]
								)],
								fluid=True
							)
		)
		return image_layout
	
	def get_horizontal_layout(self):
		image_layout =  html.Div(
							dbc.Container(
								dbc.Row(
									[
										dbc.Col(
											self.get_one_image_struct(0)
										),
										dbc.Col(
											self.get_one_image_struct(1)
										),
										dbc.Col(
											self.get_one_image_struct(2)
										),
										dbc.Col(
											self.get_one_image_struct(3)
										),
									]
								),
								fluid=True
							)
		)
		return image_layout
	
	def get_vertical_layout(self):
		image_layout =  html.Div(
							dbc.Container(
								dbc.Col(
									[
										dbc.Row(
											self.get_one_image_struct(0)
										),
										dbc.Row(
											self.get_one_image_struct(1)
										),
										dbc.Row(
											self.get_one_image_struct(2)
										),
										dbc.Row(
											self.get_one_image_struct(3)
										),
									]
								),
								fluid=True
							)
		)
		return image_layout

	def create_layout(self):
		app = self.app
		navbar = self.get_navbar()
		
		# Output
		input_layout = self.get_input_layout()
		common_navigation_button = html.Div(
							dbc.Container(
								dbc.Row(
									[
									dbc.Col(
											[
											dbc.Button(
											"<<",
											color="primary",
											className="me-1",
											id=f"All Previous Image",
											disabled=False,
										),
											dbc.Button(
												">>",
												color="primary",
												className="me-1",
												id=f"All Next Image",
												disabled=False,
											)
											]),
									dbc.Col(
										dbc.Select(
											id="View Type",
											options=self.get_options(),
											value=self.layout_type,
										)
									),
									dbc.Col([
											dbc.Input(type="number", id=f'Skip Rate', value=self.skip_rate, step=1, debounce=True),
											html.P("Skip Rate")
									], width=30
										),
									dbc.Col(
											dbc.Input(type="number", id=f'Common Nav', value=0, step=1, debounce=True), width=30
										),
									dbc.Col([
											dbc.Button(
											"Select Common",
											color="primary",
											className="me-1",
											id=f"Select Common",
											disabled=False,
										),
											dbc.Button(
											"Reset",
											color="primary",
											className="me-1",
											id=f"Reset",
											disabled=False,
										),
											dbc.Button(
											"Order_Updated",
											color="primary",
											className="me-1",
											id=f"Order_Updated",
											disabled=False,
										)
										]
									),
									dbc.Col([
											dbc.Input(type="string", id=f'Filter', debounce=True, value=self.current_filter),
											html.P(f"Filtering string")
									]),
									]
										),
								fluid=True
							))
		image_layout = self.get_image_layout()

		# for i in range(self.total_folders):
		#     self.setup_image_callback()
		image_layout = html.Div(id="image_layout", children=image_layout)

		tab0_content = dbc.Container(fluid=True, children=[
			html.H1("viewer", id="temp"),
			navbar,
			input_layout,
			common_navigation_button,
			image_layout])
		
		tab1_content = self.grid_object.make_layout()


		tabs = dbc.Tabs(
			[
				dbc.Tab(tab0_content, tab_id="tab-0", label="Image Comparer"),
				dbc.Tab(tab1_content, tab_id="tab-1", label="Thumbnailer"),
			],
			card=False,
			active_tab="tab-0",
		)


		app.layout = html.Div([tabs])

		# return children
	
	def get_options(self):
		options = [{'label': 'Grid', 'value' : 'Grid'},
				   {'label': 'Vertical', 'value' : 'Vertical'},
				   {'label': 'Horizontal', 'value' : 'Horizontal'}, ]
		return options
	
	def get_slider(self, index):
		slider = dcc.Slider(
					id=f"preview-slider{index}",
					min=0,
					max=6,
					step=None,
					marks={
						i: f"1/{v}"
						for i, v in enumerate(
							self.slider_size
						)
					},
					value=-3,
					updatemode="drag",
				)
		return slider
	
	def setup_image_callback(self, which_dir):
		app = self.app
		@app.callback(
			Output(f"main-image{which_dir}", "src"),
			Output(f"image_name{which_dir}", "value"),
			Output(f"index_counter{which_dir}", "children"),
			Input(f"preview-slider{which_dir}", "value"),
			Input(f"imagedir-input{which_dir}", "value"),
			Input(f"previous_image{which_dir}", "n_clicks"),
			Input(f"next_image{which_dir}", "n_clicks"),
			Input("All Previous Image", 'n_clicks'),
			Input("All Next Image", 'n_clicks'),
			Input(f'numnav{which_dir}','value'),
			Input('Common Nav', 'value'),
			Input("Filter", "value"),
			Input(f"Normalize Data{which_dir}", "n_clicks"),
			Input(f"Which Parition{which_dir}", "value"),
			prevent_initial_call=True,
		)
		def image_update(scale_ind, imgdir, prev_click, next_clickm, all_prev_click, all_next_clickm, numnav, comnav, filter_str, click_normalize, which_parition):
			return self.image_loader_setter(imgdir, which_dir, scale_ind, numnav, comnav, filter_str, which_parition)
	
	def setup_parition_callback(self, which_dir):
		app = self.app
		@app.callback(
			Output(f"Parition Number{which_dir}", "children"),
			Input(f"Total Paritions{which_dir}", "value"),
			prevent_initial_call=True,
		)
		def change_parition_layout(total_parition):
			changed_id = [p["prop_id"] for p in callback_context.triggered][0]
			if f"Total Paritions{which_dir}" in changed_id:
				self.total_paritions[which_dir] = int(total_parition)
				return self.parition_info(which_dir) 
			return no_update

	# Callbacks
	def setup_callback(self):
		app = self.app
		@app.callback(output=[Output(component_id="about", component_property="is_open"), 
							Output(component_id="about-popover", component_property="active")], 
					inputs=[Input(component_id="about-popover", component_property="n_clicks")], 
					state=[State("about","is_open"), State("about-popover","active")])
		def about_popover(n, is_open, active):
			if n:
				return not is_open, active
			return is_open, active
			
		for i in range(self.total_folders):
			self.setup_image_callback(i)
		
		for i in range(self.total_folders):
			self.setup_parition_callback(i)
		
		@app.callback(
			Output("Select Common", "value"),
			Input("Select Common", "n_clicks"),
			prevent_initial_call=True,
		)
		def image_selecetion(select_common):
			changed_id = [p["prop_id"] for p in callback_context.triggered][0]
			if "Select Common" in changed_id:
				self.select_common_images()
			return no_update
		
		@app.callback(
			Output(f"Skip Rate", "value"),
			Input(f"Skip Rate", "value"),
			prevent_initial_call=True,
		)
		def update_skip_rate(value):
			self.skip_rate = value
		
		@app.callback(
			Output("Reset", "value"),
			Input("Reset", "n_clicks"),
			prevent_initial_call=True,
		)
		def reset_selections(reset_click):
			changed_id = [p["prop_id"] for p in callback_context.triggered][0]
			if "Reset" in changed_id:
				self.reset_input()
			return no_update
		
		@app.callback(
			Output("Order_Updated", "value"),
			Input("Order_Updated", "n_clicks"),
			prevent_initial_call=True,
		)
		def order_updated_selections(Order_Updated_click):
			changed_id = [p["prop_id"] for p in callback_context.triggered][0]
			if "Order_Updated" in changed_id:
				self.order_updated_input()
			return no_update
		
		@app.callback(
			Output("image_layout", "children"),
			Input("View Type", "value"),
			prevent_initial_call=True,
		)
		def render_image_layout(view_type):
			changed_id = [p["prop_id"] for p in callback_context.triggered][0]
			if "View Type" in changed_id:
				self.layout_type = view_type
				return self.get_image_layout()
			return no_update
		
	def perform_filtering(self, filter_string, which_dir):
		image_lists = []
		for file in self.image_lists[which_dir]:
			if filter_string in file:
				image_lists.append(file)
		
		self.image_lists[which_dir] = image_lists
		self.indexes[which_dir] = 0
	
	def select_common_images(self):
		total_folders_for_comp = sum([elem!='' for elem in self.image_directories])
		common_files = defaultdict(lambda :0)
		for i in range(self.MAX_FOLDERS):
			for file in self.image_lists_original[i]:
				common_files[os.path.basename(file)] +=1
		
		image_lists = []
		for i in range(self.MAX_FOLDERS):
			image_lists.append([])
		for i in range(self.MAX_FOLDERS):
			for file in self.image_lists_original[i]:
				if common_files[os.path.basename(file)]==total_folders_for_comp:
					image_lists[i].append(file)
		
		self.image_lists = image_lists
		self.indexes = [0]*self.MAX_FOLDERS

	def check_correct_index(self, new_index, which_dir):
		previous_index = self.indexes[which_dir]
		if new_index>=len(self.image_lists[which_dir]):
			self.indexes[which_dir] = len(self.image_lists[which_dir])-1
		elif new_index<0:
			self.indexes[which_dir] = 0
		else:
			self.indexes[which_dir] = new_index
		return previous_index==new_index
	
	def image_loader_setter(self, imgdir, which_dir, scale_ind, numnav, comnav, filter_str, which_parition):
		image = None
		scale = self.slider_size[scale_ind]
		changed_id = [p["prop_id"] for p in callback_context.triggered][0]

		if f"imagedir-input{which_dir}" in changed_id:
			if imgdir == "":
				return no_update
			self.image_directories[which_dir] = imgdir
			self.load(imgdir, which_dir)
			image = self.read_image_convert(0, which_dir)
			self.create_layout()
		
		next_image_indicator = f'next_image{which_dir}' in changed_id or 'All Next Image' in changed_id
		if next_image_indicator and self.indexes[which_dir]<len(self.image_lists[which_dir])-1:
			is_same_index = self.check_correct_index(self.indexes[which_dir]+self.skip_rate, which_dir)
			if not is_same_index:
				image = self.read_image_convert(self.indexes[which_dir], which_dir)

		previous_image_indicator = f'previous_image{which_dir}' in changed_id or 'All Previous Image' in changed_id
		if previous_image_indicator and self.indexes[which_dir]>0:
			is_same_index = self.check_correct_index(self.indexes[which_dir]-self.skip_rate, which_dir)
			if not is_same_index:
				image = self.read_image_convert(self.indexes[which_dir], which_dir)
		
		if f'numnav{which_dir}' in changed_id:
			is_same_index = self.check_correct_index(numnav, which_dir)
			if not is_same_index:
				image = self.read_image_convert(self.indexes[which_dir], which_dir)
		
		if 'Common Nav' in changed_id:
			is_same_index = self.check_correct_index(comnav, which_dir)
			if not is_same_index:
				image = self.read_image_convert(self.indexes[which_dir], which_dir)
		
		if 'Filter' in changed_id:
			self.current_filter = filter_str
			self.perform_filtering(filter_str, which_dir)
			image = self.read_image_convert(self.indexes[which_dir], which_dir)
		
		if f"Normalize Data{which_dir}" in changed_id:
			self.normalizer[which_dir] = not self.normalizer[which_dir]
			image = self.read_image_convert(self.indexes[which_dir], which_dir)
		
		if f"Which Parition{which_dir}" in changed_id:
			self.parition_number[which_dir] = int(which_parition)

		if self.current_images[which_dir] is not None:
			self.current_scale[which_dir] = scale
			image = self.resize_convert_bytearray(scale, which_dir)
		
		if image is None:
			return no_update
		else:
			return image, os.path.basename(self.image_lists[which_dir][self.indexes[which_dir]]), f"{self.indexes[which_dir]+1}/{len(self.image_lists[which_dir])}"
	
	def resize_convert_bytearray(self, scale, which_dir):
		if self.current_images[which_dir] is not None:
			image = self.current_images[which_dir].copy()
			original_size = image.shape[:2][::-1]
			output_size = tuple([int(k // scale) for k in original_size])
			image = cv2.resize(image, output_size)
			img_h, img_w = image.shape[:2]
			image = image[:,(self.parition_number[which_dir]-1)*img_w//self.total_paritions[which_dir]:self.parition_number[which_dir]*img_w//self.total_paritions[which_dir]] 
			return "data:image/jpg;base64, " + np2base64(image, enc_format="jpeg")
		return ''
	
	def read_image_convert(self, index, which_dir):
		if len(self.image_lists[which_dir])>0 and index<=len(self.image_lists[which_dir])-1:
			image = cv2.imread(self.image_lists[which_dir][index])[..., ::-1]
			if self.normalizer[which_dir]:
				image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
			self.current_images[which_dir] = image
			img_h, img_w = image.shape[:2]
			image = image[:,(self.parition_number[which_dir]-1)*img_w//self.total_paritions[which_dir]:self.parition_number[which_dir]*img_w//self.total_paritions[which_dir]] 
			return "data:image/jpg;base64, " + np2base64(image, enc_format="jpeg")
		
		return None
	
	def load(self, imgdir, which_dir):
		images = glob.glob(imgdir)
		if len(images)==0:
			imgdir = convert_path_to_linux(imgdir)
			images = glob.glob(imgdir)
		images = sorted(images, key=lambda x:os.path.basename(x))
		self.image_lists[which_dir] = images
		self.image_lists_original[which_dir] = images.copy()
	
	def order_updated_input(self):
		for which_dir, imgdir in enumerate(self.image_directories):
			if imgdir!='':
				images = glob.glob(imgdir)
				if len(images)==0:
					imgdir = convert_path_to_linux(imgdir)
					images = glob.glob(imgdir)
				images.sort(key=os.path.getmtime)
				self.image_lists[which_dir] = images
				self.image_lists_original[which_dir] = images.copy()
				self.reset_input()

class Tab1:

	def __init__(self):
		self.image_directories = ''
		self.total_folders = 1
		self.slider_size = [4, 2, 1.5, 1.25, 1, 0.5, 0.25]
		self.current_images = {}
		self.image_list = []
		self.image_list_original = []
		self.grid_sizex = 3
		self.grid_sizey = 3
		self.current_grid_image = None
		self.current_index = 0
		self.max_index = len(self.current_images)//(self.grid_sizex*self.grid_sizey)+1
		self.current_scale = 1 
		self.current_filter = ''
		self.image_name_tag = 'basename'
		self.color_type = 'red'
		self.color_maping = self.get_color_maping()
		self.parition_number = 1
		self.total_paritions = 1
	
	def get_input_layout(self):
		input_groups = []
		input_groups.append(
			dbc.InputGroup(
				[
					dbc.Input(
						value=f"{self.image_directories}",
						size="sm",
						debounce=True,
						id=f"imagedir-input",
					)
				]
			))
		input_layout = html.Div(
			input_groups
			)
		
		return input_layout
	
	# def get_options(self):
	# 	options = [{'label': '3X3', 'value' : '3X3'},
	# 			   {'label': '2X2', 'value' : '2X2'},
	# 			   {'label': '4X4', 'value' : '4X4'},
	# 			   {'label': '16X1', 'value' : '16X1'},
	# 			   {'label': '1X16', 'value' : '1X16'}]
	# 	return options
	
	def get_options_grid(self):
		options = [{'label': '1', 'value' : '1'},
				   {'label': '2', 'value' : '2'},
				   {'label': '3', 'value' : '3'},
				   {'label': '4', 'value' : '4'},
				   {'label': '5', 'value' : '5'},
				   {'label': '6', 'value' : '6'},
				   {'label': '7', 'value' : '7'}]
		return options
	
	def get_options_name(self):
		options = [{'label': 'basename', 'value' : 'basename'},
				   {'label': 'dirname', 'value' : 'dirname'},
				   {'label': 'star', 'value' : 'star'}]
		return options
	
	def get_scale_options(self):
		options = [{'label': '1/16', 'value' : 1/16},
				   {'label': '1/8', 'value' : 1/8},
				   {'label': '1/4', 'value' : 1/4},
				   {'label': '1/2', 'value' : 1/2},
				   {'label': '1', 'value' : 1},
				   {'label': '2', 'value' : 2},
				   {'label': '3', 'value' : 3}]
		return options
	
	def get_color_options(self):
		options = [{'label': 'red', 'value' : 'red'},
				   {'label': 'blue', 'value' : 'blue'},
				   {'label': 'green', 'value' : 'green'},
				   {'label': 'black', 'value' : 'black'},
				   {'label': 'white', 'value' : 'white'}]
		return options
	
	def get_color_maping(self):
		color_code = {
			"red" : (255,0,0),
			"blue" : (0,0,255),
			"green" : (0,255,0),
			"black" : (0,0,0),
			"white" : (255,255,255)
		}
		return color_code
	
	def get_parition_options(self):
		if self.total_paritions<=1:
			options = [{'label': '1', 'value' : 1}]
		else:
			options = []
			for val in range(self.total_paritions):
				options.append({'label': f'{val+1}', 'value' : val+1})
		return options

	def parition_info(self):
		return html.Div(dbc.Select(
								id="Which Parition",
								options=self.get_parition_options(),
								value=1
							))

	def get_navigation_bar(self):
		common_navigation_button = html.Div(
									dbc.Container(
										dbc.Row(
											[
											dbc.Col(
													[
													dbc.Button(
													"<<",
													color="primary",
													className="me-1",
													id=f"Previous Set",
													disabled=False,
												),
													dbc.Button(
														">>",
														color="primary",
														className="me-1",
														id=f"Next Set",
														disabled=False,
													)
													]),
											dbc.Col([
												dbc.Select(
													id="Grid View Typex",
													options=self.get_options_grid(),
													value='3'
												),
												html.P(f"Grid Type X")
											]
											),
											dbc.Col([
												dbc.Select(
													id="Grid View Typey",
													options=self.get_options_grid(),
													value='3'
												),
												html.P(f"Grid Type Y")
											]
											),
											dbc.Col([
													dbc.Input(type="number", id=f'Navigator', value=0, step=1, debounce=True),
													html.P(id="image indexing", children=f"{self.current_index}/{len(self.image_list)}")
												],width=30
												),
											dbc.Col([
													dbc.Button(
													"Reset",
													color="primary",
													className="me-1",
													id=f"Reset State",
													disabled=False,
												)
												]
											),
											dbc.Col([
													dbc.Input(type="string", id=f'Filtering', debounce=True, value=self.current_filter),
													html.P(f"Filtering string")
											]),
											dbc.Col([
												dbc.Select(
													id="Scale Level",
													options=self.get_scale_options(),
													value=1/4
												),
												html.P(f"Scale Level")
											]
											),
											dbc.Col([
												dbc.Input(type="int", id=f'Total Paritions', debounce=True, value=self.total_paritions),
												html.P(f"Total Paritions")
											]
											),
											dbc.Col([
												html.Div(id="Parition Number", children=self.parition_info()),
												html.P(f"Parition Number")
											]
											),
											dbc.Col([
												dbc.Select(
													id="Name Color",
													options=self.get_color_options(),
													value='red'
												),
												html.P(f"Color Name")
											]
											),
											dbc.Col([
												dbc.Select(
													id="name image",
													options=self.get_options_name(),
													value='star'
												),
												html.P(f"Image Name")
											]
											),
											dbc.Col([
													dbc.Button(
													"Order_Updated",
													color="primary",
													className="me-1",
													id=f"Order_Updated_thumb",
													disabled=False,
												)
												]
											),
											dbc.Col([
													dbc.Button(
													"Download",
													color="primary",
													className="me-1",
													id=f"Download Thumbnail",
													disabled=False,
												)
												]
											),
											]
												),
										fluid=True
									))
		return common_navigation_button
	
	def make_layout(self):
		input_layout = self.get_input_layout()
		navigation_bar = self.get_navigation_bar()
		image_layout = html.Div(html.Img(id=f'main-image', src=self.current_grid_image)
		)
		layout = html.Div([
			input_layout,
			navigation_bar,
			image_layout
			]
		)
		return layout
	
	# def resize_convert_bytearray(self, scale, which_dir):
	# 	if self.current_grid_image is not None:
	# 		image = self.current_grid_image.copy()
	# 		original_size = image.shape[:2][::-1]
	# 		output_size = tuple([int(k // scale) for k in original_size])
	# 		image = cv2.resize(image, output_size)
	# 		return "data:image/jpg;base64, " + np2base64(image, enc_format="jpeg")
	# 	return ''
		
	def image_loader_setter(self, imgdir, scale_ind, comnav, grid_typex, grid_typey, image_name_type, filter_string, color_type, which_parition):
		image = None
		scale = scale_ind
		changed_id = [p["prop_id"] for p in callback_context.triggered][0]

		if f"imagedir-input" in changed_id:
			if imgdir == "":
				return no_update
			self.load(imgdir)
			image = self.read_image_convert(0)
		
		next_image_indicator = 'Next Set' in changed_id
		if next_image_indicator:
			if self.current_index+self.grid_sizex*self.grid_sizey>=len(self.image_list):
				new_index = max(0,len(self.image_list)-self.grid_sizex*self.grid_sizey)
			else:
				new_index = self.current_index+self.grid_sizex*self.grid_sizey
			if new_index!=self.current_index:
				self.current_index = new_index
				image = self.read_image_convert(self.current_index)

		previous_image_indicator = 'Previous Set' in changed_id
		if previous_image_indicator:
			if self.current_index-self.grid_sizex*self.grid_sizey<0:
				new_index = 0
			else:
				new_index = self.current_index-self.grid_sizex*self.grid_sizey
			if new_index!=self.current_index:
				self.current_index = new_index
				image = self.read_image_convert(self.current_index)
		
		condition_navigation = comnav<len(self.image_list)-1 and comnav>=0
		if 'Navigator' in changed_id and condition_navigation:
			if comnav!=self.current_index:
				self.current_index = comnav
				image = self.read_image_convert(self.current_index)
		
		if 'Grid View Typex' in changed_id or 'Grid View Typey' in changed_id:
			# self.grid_sizey, self.grid_sizex = [int(elem) for elem in grid_type.split('X')]
			self.grid_sizey = int(grid_typex)
			self.grid_sizex = int(grid_typey)
			image = self.read_image_convert(self.current_index)
		
		if 'name image' in changed_id:
			self.image_name_tag = image_name_type
			self.current_images = {}
			image = self.read_image_convert(self.current_index)
		
		if "Filtering" in changed_id:
			self.current_filter = filter_string
			self.perform_filtering(filter_string)
			image = self.read_image_convert(self.current_index)
		
		if 'Name Color' in changed_id:
			self.color_type = color_type
			image = self.read_image_convert(self.current_index)

		if  'Which Parition' in changed_id:
			self.parition_number = int(which_parition)
			image = self.read_image_convert(self.current_index)

		if self.current_grid_image is not None:
			self.current_scale = float(scale)
			image = self.resize_convert_bytearray()
		
		if image is None:
			return no_update
		else:
			return image, f"{self.current_index+1}/{len(self.image_list)}"

	def setup_image_callback(self, app):
		@app.callback(
			Output(f"main-image", "src"),
			Output("image indexing", "children"),
			Input(f"Scale Level", "value"),
			Input(f"imagedir-input", "value"),
			Input("Previous Set", 'n_clicks'),
			Input("Next Set", 'n_clicks'),
			Input('Navigator', 'value'),
			Input('Grid View Typex', 'value'),
			Input('Grid View Typey', 'value'),
			Input('name image','value'),
			Input("Filtering", "value"),
			Input("Name Color", "value"),
			Input("Which Parition", "value"),
			prevent_initial_call=True
		)
		def image_update(scale_ind, imgdir, all_prev_click, all_next_clickm, comnav, grid_typex, grid_typey, image_name_type, filter_str, color_type, which_parition):
			return self.image_loader_setter(imgdir, scale_ind, comnav, grid_typex, grid_typey, image_name_type, filter_str, color_type, which_parition)
	

	# Callbacks
	def setup_callback(self, app):

		self.setup_image_callback(app)

		@app.callback(
			Output("Download Thumbnail",'n_clicks'),
			Input("Download Thumbnail", "n_clicks"),
			prevent_initial_call=True,
		)
		def download_thimbnail(download_click):
			if self.current_grid_image is not None:
				potential_folder_location = os.path.dirname(self.image_directories[:self.image_directories.find('*')])
				if not os.path.exists(potential_folder_location):
					potential_folder_location = os.path.dirname(potential_folder_location)
				if os.path.exists(potential_folder_location):
					cv2.imwrite(os.path.join(potential_folder_location,f"{int(datetime.utcnow().timestamp())}.jpg"),self.return_resized_image()[...,::-1])

		@app.callback(
			Output("Reset State", "value"),
			Input("Reset State", "n_clicks"),
			prevent_initial_call=True,
		)
		def reset_selections(reset_click):
			changed_id = [p["prop_id"] for p in callback_context.triggered][0]
			if "Reset State" in changed_id:
				self.reset_input()
			return no_update
		
		@app.callback(
			Output("Parition Number", "children"),
			Input("Total Paritions", "value"),
			prevent_initial_call=True,
		)
		def change_parition_layout(total_parition):
			changed_id = [p["prop_id"] for p in callback_context.triggered][0]
			if "Total Paritions" in changed_id:
				self.total_paritions = int(total_parition)
				return self.parition_info() 
			return no_update
		
		@app.callback(
			Output("Order_Updated_thumb", "value"),
			Input("Order_Updated_thumb", "n_clicks"),
			prevent_initial_call=True,
		)
		def order_updated_thumb_selections(Order_Updated_thumb_click):
			changed_id = [p["prop_id"] for p in callback_context.triggered][0]
			if "Order_Updated_thumb" in changed_id:
				self.order_updated_input()
			return no_update
		
	def perform_filtering(self, filter_string):
		image_list = []
		for file in self.image_list:
			if filter_string in file:
				image_list.append(file)
		
		self.image_list = image_list
		self.current_index = 0
	
	def reset_input(self):
		self.image_list = self.image_list_original.copy()
		self.current_index = 0
	
	def load(self, imgdir):
		images = glob.glob(imgdir)
		if len(images)==0:
			imgdir = convert_path_to_linux(imgdir)
			images = glob.glob(imgdir)
		images = sorted(images, key=lambda x:os.path.basename(x))
		self.image_directories = imgdir
		self.image_list = images
		self.image_list_original = images.copy()
	
	def create_grid(self, scale=1):
		output_image = None
		for i,file_name in enumerate(self.current_images):
			image_to_put = self.current_images[file_name]
			if output_image is None:
				height, width = image_to_put.shape[:2]
				width = int(width/self.total_paritions)
				output_image = np.zeros([int(height*self.grid_sizey*scale), int(width*self.grid_sizex*scale), 3], dtype=np.uint8)
			image_to_put = image_to_put[:,(self.parition_number-1)*width:(self.parition_number)*width]
			starting_height_main_image = int(height*scale)*(i//self.grid_sizex)
			starting_width = (int(width*scale))*(i%self.grid_sizex)
			if self.image_name_tag=='basename':
				file_name_use = os.path.basename(file_name)
			elif self.image_name_tag=='dirname':
				file_name_use = os.path.basename(os.path.dirname(file_name))
			else:
				starting_index = self.image_directories.find('*')
				file_name_use = file_name[starting_index:]
				# if not os.path.exists(file_name_use):
				# 	file_name_use = os.path.dirname(file_name_use)
				# file_name_use = os.path.basename(file_name_use)
			image_to_put = self.resize_image(image_to_put, scale)
			image_to_put = self.overlay_text(image_to_put.copy(), file_name_use, self.color_maping[self.color_type])
			height_one, width_one = image_to_put.shape[:2]
			output_image[starting_height_main_image:starting_height_main_image+height_one,starting_width:width_one+starting_width] = image_to_put
		return output_image
	
	@staticmethod
	def overlay_text(img, file_name, color_code):
		font                   = cv2.FONT_HERSHEY_SIMPLEX
		bottomLeftCornerOfText = (10,img.shape[0]-10)
		fontScale              = round(0.25*img.shape[0]/150,2)
		fontColor              = color_code
		thickness              = int(1*img.shape[0]/150)
		lineType               = 2

		img = cv2.putText(img,f'{file_name}', 
				bottomLeftCornerOfText, 
				font, 
				fontScale,
				fontColor,
				thickness,
				lineType)
		return img


	def read_image_convert(self, index):
		temp_dict = {}
		image_size = None
		for i,file_name in enumerate(self.image_list[index:index+self.grid_sizex*self.grid_sizey]):
			if file_name in self.current_images:
				temp_dict[file_name] = self.current_images[file_name]
			else:
				temp_dict[file_name] = cv2.imread(self.image_list[index+i])[..., ::-1]
				# temp_dict[file_name] = image
			if image_size is None:
				image_size = temp_dict[file_name].shape[:2]
		self.current_images = temp_dict
		if image_size is not None:
			self.current_grid_image = self.create_grid()
			return "data:image/jpg;base64, " + np2base64(self.current_grid_image, enc_format="jpeg")
		else:
			return ""
		
	def resize_image(self, image, scale):
		original_size = image.shape[:2][::-1]
		output_size = tuple([int(k*scale) for k in original_size])
		image = cv2.resize(image, output_size)
		return image
	
	def return_resized_image(self):
		scale = self.current_scale
		image = self.current_grid_image.copy()
		image = self.resize_image(image, scale)
		return image
	
	def resize_convert_bytearray(self):
		if self.current_grid_image is not None:
			scale = self.current_scale
			image = self.create_grid(scale)
			return "data:image/jpg;base64, " + np2base64(image, enc_format="jpeg")
		return ''
	
	def order_updated_input(self):
		imgdir = self.image_directories
		if imgdir!='':
			images = glob.glob(imgdir)
			if len(images)==0:
				imgdir = convert_path_to_linux(imgdir)
				images = glob.glob(imgdir)
			images.sort(key=os.path.getmtime)
			self.image_list = images
			self.image_list_original = images.copy()
			self.reset_input()

if __name__ == "__main__":
	image_viewer = Viewer()
	image_viewer.setup()
	image_viewer.run()