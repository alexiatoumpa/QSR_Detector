#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""QSTAR Utilities
"""
from __future__ import print_function
from itertools import combinations, permutations
import copy, sys
from igraph import Graph as iGraph
import numpy as np
import warnings
import qstag



def compute_episodes(world_qsr):
	"""
	Compute QSR Episodes from a QSRLib.QSR_World_Trace object.
	QSR Episodes compresses repeating QSRs into a temporal interval over which they hold.
	Returns: a long list of episodes with the format, `[(objects), {spatial relations}, (start_frame, end_frame)]`.

	FILTERS: if any of the qsr values == Ignore, the entire episode will be ignored.

	Example content:
	----------------
	o1,mug,o2,hand,sur,3,7
	o1,mug,o3,head,con,4,9
	o2,hand,o3,head,dis,1,9
	----------------
	..seealso:: For further details about QSR Episodes, refer to its :doc:`description. <../handwritten/qsrs/qstag/>`

	:param world_qsr: The QSR_World_Trace object (QSRlib_Response_Message)
	:type world_qsr: :class:`World_QSR_Trace <qsrlib_io.world_qsr_trace>`
	"""

	episodes = []
	obj_based_qsr_world = {}
	frames = world_qsr.get_sorted_timestamps()

	"""remove the first frame which cannot contain a qtcb relation"""
	if "qtcbs" in world_qsr.qsr_type:
		if frames[0] == 1.0: frames.pop(0)

	for frame in frames:
		for objs, qsrs in world_qsr.trace[frame].qsrs.items():
			my_qsrs = {}
			#print("h", objs, qsrs.qsr)

			for qsr_key, qsr_val in qsrs.qsr.items():
				#print("  ", qsr_key, qsr_val)
				if qsr_key is "tpcc":
					origin,relatum,datum = objs.split(',')
					new_key=("%s-%s,%s") % (origin,relatum,datum)
					try:
						obj_based_qsr_world[new_key].append((frame, {"tpcc": qsrs.qsr["tpcc"]}))
					except KeyError:
						obj_based_qsr_world[new_key] = [(frame, {"tpcc": qsrs.qsr["tpcc"]})]
				else:
					my_qsrs[qsr_key] = qsr_val

			if my_qsrs != {}:
				try:
					obj_based_qsr_world[objs].append((frame, my_qsrs))
				except KeyError:
					obj_based_qsr_world[objs] = [(frame, my_qsrs)]

	#print("s", obj_based_qsr_world[objs])
	for objs, frame_tuples in obj_based_qsr_world.items():
		epi_start, epi_rel = frame_tuples[0]
		epi_end  = copy.copy(epi_start)

		objects = objs.split(',')
		for (frame, rel) in frame_tuples:
			if rel == epi_rel:
				epi_end = frame
			else:
				episodes.append( (objects, epi_rel, (epi_start, epi_end)) )
				epi_start = epi_end = frame
				epi_rel = rel
		episodes.append((objects, epi_rel, (epi_start, epi_end)))

	"""If any of the qsr values == ignore. Remove that episode entirely. """
	filtered_out_ignore = []
	for ep in episodes:
		ignore_flag = 0
		for qsr, val in ep[1].items():
			if val == "Ignore":	ignore_flag = 1
		if ignore_flag == 0: filtered_out_ignore.append(ep)

	#print("number of eps:", len(filtered_out_ignore)) # MY CODE: COMMENTED IT OUT

	return filtered_out_ignore

def get_E_set(objects, spatial_data):
	"""Returns the Starting episode set (E_s) and the Endding episode set (E_s)
	See Sridar_AAAI_2010 for more details

	:param objects: object dictionary with name as key, and node ID as value
	:type objects: dictionary
	:param spatial_data: A list of tuples, where a tuple contains a list of objects, a spatial relation node ID, and a duration of time.
	:type spatial_data: list
	:return: A tuple containing two sets of QSR Episodes, where a temporal node does not hold beteen Episodes in the same set.
	:rtype: tuple
	"""
	objects_ids = objects.values()
	start, end = {}, {}
	E_s, E_f = [], []
	number_of_objects = len(spatial_data[0][0])

	for possible_ids in permutations(objects_ids, number_of_objects):
		added=0
		for epi in spatial_data:
			ep_objects =  epi[0]
			frame_window = epi[2]

			#if (objects[0] == obj1 and objects[1] == obj2):
			if list(possible_ids) == ep_objects:
				start[frame_window[0]] = epi
				end[frame_window[1]] = epi
				added=1
		if added == 1:
			st=start.keys()
			st.sort()
			E_s.append(start[st[0]])
			en=end.keys()
			en.sort()
			E_f.append(end[en[-1]])
	return E_s, E_f

def get_allen_relation(duration1, duration2):
	"""Generates an Allen interval algebra relation between two discrete durations of time

	:param duration1: First duration of time (start_frame, end_frame)
	:type duration1: tuple
	:param duration2: Second duration of time (start_frame, end_frame)
	:type duration2: tuple
	"""

	is1, ie1 = duration1
	is2, ie2 = duration2

	if is2-1 == ie1:
		return 'm'
	elif is1-1 == ie2:
		return 'mi'

#	elif is1 == is2 and ie1 == ie2:
#		return '='

	elif is2 > ie1:
		return '<'
	elif is1 > ie2:
		return '>'

	### I INCLUDED THIS !!
	else: 
		return 'o'

#	elif ie1 >= is2 and ie1 < ie2 and is1 < is2:
#		return 'o'
#	elif ie2 >= is1 and ie2 < ie1 and is2 < is1:
#		return 'oi'
#	elif is1 > is2 and ie1 < ie2:
#		return 'd'
#	elif is1 < is2 and ie1 > ie2:
#		return 'di'
#	elif is1 == is2 and ie1 < ie2:
#		return 's'
#	elif is1 == is2 and ie1 > ie2:
#		return 'si'
#	elif ie1 == ie2 and is2 < is1:
#		return 'f'
#	elif ie1 == ie2 and is2 > is1:
#		return 'fi'


def graph_hash(G, node_name_attribute='name', edge_name_attribute=None):
	"""
	See Figure 4 in 'kLog: A Language for Logical and Relational Learning with Kernels'
	for the algorithm.

	Takes an igraph graph, node_name attribute and edge_name_attribute. Note that
	edge_name_attribute is optional i.e. for graphs without edge labels or to ignore edge labels,
	edge_name_attribute is None.
	"""

	# suppress Runtime Warnings regarding not being able to find a path through the graphs
	warnings.filterwarnings('ignore')

	for node in G.vs:
		paths = G.get_shortest_paths(node)
		node_hashes = []
		for path in paths:
			if len(path) != 0:
				node_name = G.vs[path[-1]][node_name_attribute]
				if node_name == None:
					node_name = repr(None)
				node_hashes.append((len(path), node_name))
		node_hashes.sort()
		node_hashes_string = ':'.join([repr(i) for i in node_hashes])
		node['hash_name'] = hash(node_hashes_string)
	warnings.filterwarnings('always')
	if edge_name_attribute:
		edge_hashes = [(G.vs[edge.source]['hash_name'], G.vs[edge.target]['hash_name'],\
								   edge[edge_name_attribute]) for edge in G.es]
	else:
		edge_hashes = [(G.vs[edge.source]['hash_name'], G.vs[edge.target]['hash_name'])\
					   for edge in G.es]
	edge_hashes.sort()
	edge_hashes_string = ':'.join([repr(i) for i in edge_hashes])
	return hash(edge_hashes_string)

def get_temporal_chords_from_episodes(episodes):
	"""
	Function returns temporal chords from a subset of episodes

	:param episodes: a list of episodes, where one epiode has the format (start_frame, end_frame, id)
	:type episodes: list
	:return: list of chords
	:rtype: list
	"""
	interval_data = {}
	interval_breaks = []
	# For each time point in the combined interval, get the state of the
	# system which is just a list of relations active in that time point.

	#todo: can this work with floats? Not unless there is a measure of unit.
	for (s, e, id_) in episodes:
		for i in range(int(s), int(e+1)):
			if i not in interval_data:
				interval_data[i] = []
			interval_data[i].append(id_)

	keys = interval_data.keys()
	keys.sort()

	# Now based on the state changes, break the combined interval
	# whenever there is a change in the state
	start = keys[0]
	interval_value = interval_data[start]
	for i in keys:
		if interval_value == interval_data[i]:
			end = i
			continue
		else:
			interval_breaks.append([start, end, interval_value])
			start = i
			end = i
			interval_value = interval_data[start]
	else:
		# Adding the final interval
		interval_breaks.append([start, end, interval_value])
	return interval_breaks



# -------------------------- MY CODE - MY FUNCTIONS -------------------------- #
def color_temporal_nodes(graph, dot_file):

    # TKinter Color Chart: http://www.science.smith.edu/dftwiki/index.php/Color_Charts_for_TKinter
    COLORS = ['snow', 'ghostwhite', 'whitesmoke', 'gainsboro', 'floralwhite',
    'oldlace', 'linen', 'antiquewhite', 'papayawhip', 'blanchedalmond', 'bisque',
    'peachpuff', 'navajowhite', 'lemonchiffon', 'mintream', 'azure', 'aliceblue',
    'lavender', 'lavenderblush', 'mistyrose', 'darkslategray', 'dimgray', 'slategray',
    'lightslategray', 'gray', 'lightgrey', 'midnightblue', 'navy', 'cornflowerblue',
    'darkslateblue', 'slateblue', 'mediumslateblue', 'lightslateblue', 'mediumblue',
    'royalblue',  'blue', 'dodgerblue', 'deepskyblue', 'skyblue', 'lightskyblue',
    'steelblue', 'lightsteelblue', 'lightblue', 'powderblue', 'paleturquoise',
    'darkturquoise', 'mediumturquoise', 'turquoise', 'cyan', 'lightcyan', 'cadetblue',
    'mediumaquamarine', 'aquamarine', 'darkgreen', 'darkolive green', 'darkseagreen',
    'seagreen', 'mediumseagreen', 'lightseagreen', 'palegreen', 'springgreen',
    'lawngreen', 'mediumspringgreen', 'greenyellow', 'limegreen', 'yellowgreen',
    'forestgreen', 'olivedrab', 'darkkhaki', 'khaki', 'palegoldenrod', 'lightgoldenrodyellow',
    'lightyellow', 'yellow', 'gold', 'lightgoldenrod', 'goldenrod', 'darkgoldenrod',
    'rosybrown', 'indianred', 'saddlebrown', 'sandybrown', 'darksalmon', 'salmon',
    'lightsalmon', 'orange', 'darkorange', 'coral', 'lightcoral', 'tomato',
    'orangered', 'red', 'hotpink', 'deeppink', 'pink', 'lightpink', 'palevioletred',
    'maroon', 'mediumvioletred', 'violetred', 'mediumorchid', 'darkorchid',
    'darkviolet', 'blueviolet', 'purple', 'mediumpurple', 'thistle', 'snow2',
    'snow3', 'snow4', 'seashell2', 'seashell3', 'seashell4', 'AntiqueWhite1',
    'AntiqueWhite2', 'AntiqueWhite3', 'AntiqueWhite4', 'bisque2', 'bisque3',
    'bisque4', 'PeachPuff2', 'PeachPuff3', 'PeachPuff4', 'NavajoWhite2', 'NavajoWhite3',
    'NavajoWhite4', 'LemonChiffon2', 'LemonChiffon3', 'LemonChiffon4', 'cornsilk2',
    'cornsilk3', 'cornsilk4', 'ivory2', 'ivory3', 'ivory4', 'honeydew2', 'honeydew3',
    'honeydew4', 'LavenderBlush2', 'LavenderBlush3', 'LavenderBlush4', 'MistyRose2',
    'MistyRose3', 'MistyRose4', 'azure2', 'azure3', 'azure4', 'SlateBlue1',
    'SlateBlue2', 'SlateBlue3', 'SlateBlue4', 'RoyalBlue1', 'RoyalBlue2', 'RoyalBlue3',
    'RoyalBlue4', 'blue2', 'blue4', 'DodgerBlue2', 'DodgerBlue3', 'DodgerBlue4',
    'SteelBlue1', 'SteelBlue2', 'SteelBlue3', 'SteelBlue4', 'DeepSkyBlue2',
    'DeepSkyBlue3', 'DeepSkyBlue4', 'SkyBlue1', 'SkyBlue2', 'SkyBlue3', 'SkyBlue4',
    'LightSkyBlue1', 'LightSkyBlue2', 'LightSkyBlue3', 'LightSkyBlue4', 'SlateGray1',
    'SlateGray2', 'SlateGray3', 'SlateGray4', 'LightSteelBlue1', 'LightSteelBlue2',
    'LightSteelBlue3', 'LightSteelBlue4', 'LightBlue1', 'LightBlue2', 'LightBlue3',
    'LightBlue4', 'LightCyan2', 'LightCyan3', 'LightCyan4', 'PaleTurquoise1',
    'PaleTurquoise2', 'PaleTurquoise3', 'PaleTurquoise4', 'CadetBlue1', 'CadetBlue2',
    'CadetBlue3', 'CadetBlue4', 'turquoise1', 'turquoise2', 'turquoise3', 'turquoise4',
    'cyan2', 'cyan3', 'cyan4', 'DarkSlateGray1', 'DarkSlateGray2', 'DarkSlateGray3',
    'DarkSlateGray4', 'aquamarine2', 'aquamarine4', 'DarkSeaGreen1', 'DarkSeaGreen2',
    'DarkSeaGreen3', 'DarkSeaGreen4', 'SeaGreen1', 'SeaGreen2', 'SeaGreen3',
    'PaleGreen1', 'PaleGreen2', 'PaleGreen3', 'PaleGreen4', 'SpringGreen2',
    'SpringGreen3', 'SpringGreen4', 'green2', 'green3', 'green4', 'chartreuse2',
    'chartreuse3', 'chartreuse4', 'OliveDrab1', 'OliveDrab2', 'OliveDrab4',
    'DarkOliveGreen1', 'DarkOliveGreen2', 'DarkOliveGreen3', 'DarkOliveGreen4',
    'khaki1', 'khaki2', 'khaki3', 'khaki4', 'LightGoldenrod1', 'LightGoldenrod2',
    'LightGoldenrod3', 'LightGoldenrod4', 'LightYellow2', 'LightYellow3', 'LightYellow4',
    'yellow2', 'yellow3', 'yellow4', 'gold2', 'gold3', 'gold4', 'goldenrod1',
    'goldenrod2', 'goldenrod3', 'goldenrod4', 'DarkGoldenrod1', 'DarkGoldenrod2',
    'DarkGoldenrod3', 'DarkGoldenrod4', 'RosyBrown1', 'RosyBrown2', 'RosyBrown3',
    'RosyBrown4', 'IndianRed1', 'IndianRed2', 'IndianRed3', 'IndianRed4', 'sienna1',
    'sienna2', 'sienna3', 'sienna4', 'burlywood1', 'burlywood2', 'burlywood3',
    'burlywood4', 'wheat1', 'wheat2', 'wheat3', 'wheat4', 'tan1', 'tan2', 'tan4',
    'chocolate1', 'chocolate2', 'chocolate3', 'firebrick1', 'firebrick2', 'firebrick3',
    'firebrick4', 'brown1', 'brown2', 'brown3', 'brown4', 'salmon1', 'salmon2',
    'salmon3', 'salmon4', 'LightSalmon2', 'LightSalmon3', 'LightSalmon4', 'orange2',
    'orange3', 'orange4', 'DarkOrange1', 'DarkOrange2', 'DarkOrange3', 'DarkOrange4',
    'coral1', 'coral2', 'coral3', 'coral4', 'tomato2', 'tomato3', 'tomato4', 'OrangeRed2',
    'OrangeRed3', 'OrangeRed4', 'red2', 'red3', 'red4', 'DeepPink2', 'DeepPink3',
    'DeepPink4', 'HotPink1', 'HotPink2', 'HotPink3', 'HotPink4', 'pink1', 'pink2',
    'pink3', 'pink4', 'LightPink1', 'LightPink2', 'LightPink3', 'LightPink4',
    'PaleVioletRed1', 'PaleVioletRed2', 'PaleVioletRed3', 'PaleVioletRed4',
    'maroon1', 'maroon2', 'maroon3', 'maroon4', 'VioletRed1', 'VioletRed2',
    'VioletRed3', 'VioletRed4', 'magenta2', 'magenta3', 'magenta4', 'orchid1',
    'orchid2', 'orchid3', 'orchid4', 'plum1', 'plum2', 'plum3', 'plum4', 'MediumOrchid1',
    'MediumOrchid2', 'MediumOrchid3', 'MediumOrchid4', 'DarkOrchid1', 'DarkOrchid2',
    'DarkOrchid3', 'DarkOrchid4', 'purple1', 'purple2', 'purple3', 'purple4',
    'MediumPurple1', 'MediumPurple2', 'MediumPurple3', 'MediumPurple4', 'thistle1',
    'thistle2', 'thistle3', 'thistle4', 'gray1', 'gray2', 'gray3', 'gray4', 'gray5',
    'gray6', 'gray7', 'gray8', 'gray9', 'gray10', 'gray11', 'gray12', 'gray13',
    'gray14', 'gray15', 'gray16', 'gray17', 'gray18', 'gray19', 'gray20', 'gray21',
    'gray22', 'gray23', 'gray24', 'gray25', 'gray26', 'gray27', 'gray28', 'gray29',
    'gray30', 'gray31', 'gray32', 'gray33', 'gray34', 'gray35', 'gray36', 'gray37',
    'gray38', 'gray39', 'gray40', 'gray42', 'gray43', 'gray44', 'gray45', 'gray46',
    'gray47', 'gray48', 'gray49', 'gray50', 'gray51', 'gray52', 'gray53', 'gray54',
    'gray55', 'gray56', 'gray57', 'gray58', 'gray59', 'gray60', 'gray61', 'gray62',
    'gray63', 'gray64', 'gray65', 'gray66', 'gray67', 'gray68', 'gray69', 'gray70',
    'gray71', 'gray72', 'gray73', 'gray74', 'gray75', 'gray76', 'gray77', 'gray78',
    'gray79', 'gray80', 'gray81', 'gray82', 'gray83', 'gray84', 'gray85', 'gray86',
    'gray87', 'gray88', 'gray89', 'gray90', 'gray91', 'gray92', 'gray93', 'gray94',
    'gray95', 'gray97', 'gray98', 'gray99']

    # Find unique temporal relations.
    temp_list = []
    for o in graph.temporal_nodes:
        temp_list.append(o['name'])
    unique_temporal,unique_temp_counts = np.unique(temp_list, return_counts = True)
	
    # Manager the color distribution depending on the unique temporal relations captured.
    color = 40
    which_color = []
    for i in range(len(unique_temporal)):
        which_color.append(COLORS[color])
        color += 4

    # Assign color to every node depending on its value.
    for tnode in graph.temporal_nodes:
        for l in range(len(unique_temporal)):
            if tnode['name'] == unique_temporal[l]:
                tnode_name2print = tnode['name'][:-12]+ '\n' + tnode['name'][-12:] 
                if tnode_name2print[:1] == 'o':
                    tnode_name2print = '\xc3\xb5' + tnode_name2print[1:] # õ: \xc3\xb5
                dot_file.write('	%s [fillcolor=%s, label="%s", shape=ellipse];\n' %(tnode.index, which_color[l], tnode_name2print))

    return graph, dot_file



def graph2dot(graph, out_dot_file):
	"""To visualize the iGraph graph, this prints a dot file to the file location given

	:param graph: Activity Graph object (QSTAG)
	:type graph: Activity_Graph type
	:param out_dot_file: file location to save the dot file
	:type out_dot_file: string
	"""

	# Write the graph to dot file
	# Can generate a graph figure from this .dot file using the 'dot' command
	# dot -Tpng input.dot -o output.png
	dot_file = open(out_dot_file, 'w')
	dot_file.write('digraph activity_graph {\n')
	dot_file.write('	size = "45,45";\n')
	dot_file.write('	node [fontsize = "18", shape = "box", style="filled", fillcolor="aquamarine"];\n')
	dot_file.write('	ranksep=5;\n')
	# Create temporal nodes
	dot_file.write('	subgraph _1 {\n')
	dot_file.write('	rank="source";\n')

	#print(graph.temporal_nodes)
	#print(graph.spatial_nodes)
	#print(graph.object_nodes)
        #print(qstag.spatial_obj_final_edges, qstag.temp_spatial_final_edges)


        graph, dot_file = color_temporal_nodes(graph, dot_file)
	# I COMMENTED THESE COUPLE OF LINES
	#for tnode in graph.temporal_nodes:
	#	dot_file.write('	%s [fillcolor=%s, label="%s", shape=ellipse];\n' %(tnode.index, "write", tnode['name']))
	dot_file.write('}\n')

	# Create spatial nodes
	dot_file.write('	subgraph _2 {\n')
	dot_file.write('	rank="same";\n')
	for rnode in graph.spatial_nodes:
		dot_file.write('	%s [fillcolor="lightblue", label="%s"];\n' %(rnode.index, rnode['name']))
	dot_file.write('}\n')

	# Create object nodes
	dot_file.write('	subgraph _3 {\n')
	dot_file.write('	rank="sink";\n')
	for onode in graph.object_nodes:
		dot_file.write('%s [fillcolor="tan1", label="%s"];\n' %(onode.index, onode['name']))
	dot_file.write('}\n')

	# Create temporal to spatial edges
	for t_edge in graph.temp_spatial_edges:
		dot_file.write('%s -> %s [arrowhead = "normal", color="red"];\n' %(t_edge[0], t_edge[1]))

	# Create spatial to object edges
	for r_edge in graph.spatial_obj_edges:
		dot_file.write('%s -> %s [arrowhead = "normal", color="red"];\n' %(r_edge[0], r_edge[1]))
	dot_file.write('}\n')
	dot_file.close()
