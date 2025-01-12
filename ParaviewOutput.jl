__precompile__()

module ParaviewOutput

export output_mesh, output_node_sdata, output_node_vdata,
	output_element_sdata, write_outputfile

import WriteVTK	
abstract type AbstractElement end   # This is a type of forward delcaration
									# till I get this set up from FEM.jl
#-----------------------------------------------------------------------
topovtktype = Dict("Line"=>WriteVTK.VTKCellTypes.VTK_LINE,
                    "Tria"=>WriteVTK.VTKCellTypes.VTK_TRIANGLE,
					"Quad"=>WriteVTK.VTKCellTypes.VTK_QUAD,
				 	"Tetra"=>WriteVTK.VTKCellTypes.VTK_TETRA,
					"Hexa"=>WriteVTK.VTKCellTypes.VTK_HEXAHEDRON,
					"Penta"=>WriteVTK.VTKCellTypes.VTK_PENTAGONAL_PRISM)

#elemvtktype(e::AbstractElement) = topovtktype[elemtopo(e)]
#= function output_mesh(filename, node, element::AbstractElement)
	"""
	Writes a finite element type data to a vtk file opbject.  Note this does not actually
	write the file yet.  This is done in write_outputfile().

	output_mesh(filename, node, element)
	
	filename = output file name
	node = node coordinate array
	element is an AbstractElement type
	"""
	ne = numelem(element)
    cell =  Array{WriteVTK.MeshCell,1}(undef,ne)
    for e=1:ne
      cell[e] = WriteVTK.MeshCell(elemvtktype(element), element.conn[:,e])
    end
    vtkfile = WriteVTK.vtk_grid(filename, node, cell)
	return vtkfile
end =#

function output_mesh(filename, node, conn::AbstractArray, etype)
	"""	
	Writes a finite element type data to a vtk file opbject.  Note this does not actually
	write the file yet.  This is done in write_outputfile().

	output_mesh(filename, node, conn, etype)
	
	filename = output file name
	node = node coordinate array
	conn = connectivity array``
	etype = {"Line", "Tria", "Quad", "Tetra", "Hexa", "Penta"}
	"""
	nne, ne = size(conn)
	vtktype = topovtktype[etype]
    cell =  Array{WriteVTK.MeshCell,1}(undef,ne)
    for e=1:ne
      cell[e] = WriteVTK.MeshCell(vtktype, conn[:,e])
    end
    vtkfile = WriteVTK.vtk_grid(filename, node, cell)
	return vtkfile
end

function output_node_sdata(vtkfile, d, dataname)
	"""
	writes scalar nodal data to a vtk file opbject.  Note this does not actually
	write the file yet.  This is done in write_outputfile().
	"""
	WriteVTK.vtk_point_data(vtkfile, d, dataname)
end

function output_node_vdata(vtkfile, d, dataname, ddofs=[1,2,3], stride=length(ddofs))
	"""
	writes vector nodal data to a vtk file opbject.  Note this does not actually
	write the file yet.  This is done in write_outputfile().

	vtkfile a vtk output object. Typically comes from output_mesh()
	d nx1 array of nodal data
	dataname string of the data name
	ddofs vector of nodal dofs, default is [1,2,3]
	stride stride between next node data, default is length(ddofs)
	"""
	# put d vector data in sdim x numnode array form
	nn = Int(floor(length(d)/stride))
	ns = length(ddofs)
	dv = reshape(d, (stride,nn))
	WriteVTK.vtk_point_data(vtkfile, dv[ddofs,:], dataname)
end

function output_element_sdata(vtkfile, d, dataname)
	"""
	writes vector nodal data to a vtk file opbject.  Note this does not actually
	write the file yet.  This is done in write_outputfile().

	vtkfile a vtk output object. Typically comes from output_mesh()
	d is a nex1 array of scalar element data
	dataname string of the data name
	"""
	WriteVTK.vtk_cell_data(vtkfile, d, dataname)
end

function write_outputfile(vtkfile)
	"""
	Writes the data to a file from a vtk object.
	
	function write_outputfile(vtkfile)
	"""
	outfiles = WriteVTK.vtk_save(vtkfile)
end

end
# -------------------- END OF ParaviewOutput MODULE -------------------#