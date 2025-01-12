#__precompile__()

module NastranInput

export read_mesh, read_forces, read_forces!, read_spcs, read_spcs!
export ForceCard, SpcCard

using FiniteMesh: Mesh  
# https://juliapackages.com/p/finitemesh
using FemBasics: REALTYPE, IDTYPE

NastranEtype = Dict("triangle"=>"Tria3", "tria6"=>"Tria6",
                    "quad"=>"Quad4", "quad8"=>"Quad8")

function read_mesh(filename)
    mesh = Mesh(filename)
    nodes = mesh.points'
    elements = Dict{String, Matrix{IDTYPE}}()
    for (i, et) in enumerate(mesh.cells.type)
        elements[NastranEtype[et]] = (mesh.cells.index[i])'
    end
    return nodes, elements
end

struct FiniteElementRegion
    id::IDTYPE
    pid::IDTYPE
    mid::IDTYPE
    etype::String
    conn::Array{IDTYPE,2}
end

function read_regions(filename)
    mesh = Mesh(filename)
    nodes = mesh.points'
    

    return nodes
end

padline(line, n=80) = rpad(line, n)

function getid(str)
    id = tryparse(IDTYPE, str)
    if isnothing(id)
        id = 0
    end
    return id
end

function getnum(str)
    s = tryparse(REALTYPE, str)
    if isnothing(s)
        s = 0.0
    end
    return s
end

function getids(ids::String)
    dofs = replace(ids, " " => "")
    #if length(dofs)<2   # returns an integer if there is a single id
    #    return getid(ids)
    #end
    iis = Array{IDTYPE,1}(undef,length(dofs))
    for (i,c) in enumerate(dofs)
        iis[i] = tryparse(IDTYPE,string(c))
    end
    return iis
end

struct ForceCard
    sid::IDTYPE
    nid::IDTYPE
    vec::Array{REALTYPE,1}
end

function numcards(filename, card)
    f = open(filename, "r")
    cl = length(card)
    n = 0
    for lines in readlines(f)
        line = padline(lines, cl)
        if line[1:cl] == card
            n += 1
        end
    end
    close(f)
    return n
end

function read_forces(filename)
    f = open(filename, "r")
    nf = numcards(filename, "FORCE")
    forces = Array{ForceCard,1}(undef,nf)
    i = 0
    for lines in readlines(f)
        line = padline(lines)
        if line[1:5]=="FORCE"
            sid = getid(line[9:16])
            nid = getid(line[17:24])
            mag = getnum(line[33:40])
            n1  = getnum(line[41:48])
            n2  = getnum(line[49:56])
            n3  = getnum(line[57:64])
            i += 1
            forces[i] = ForceCard(sid, nid, mag*[n1,n2,n3])
        end
    end
    close(f)
    return forces
end

function read_forces!(filename, fext, ndpn=6)
    f = open(filename, "r")
    for lines in readlines(f)
        line = padline(lines)
        if line[1:5]=="FORCE"
            sid = getid(line[9:16])
            nid = getid(line[17:24])
            mag = getnum(line[33:40])
            n1  = getnum(line[41:48])
            n2  = getnum(line[49:56])
            n3  = getnum(line[57:64])
            i = ndpn*(nid-1)
            fext[i+1] += mag*n1
            fext[i+2] += mag*n2 
            fext[i+3] += mag*n3 
        end
    end
    close(f)
end

struct SpcCard
    sid::IDTYPE
    nid::IDTYPE
    ifix::Array{IDTYPE,1}
    ival::REALTYPE
end

function read_spcs(filename)
    nf = numcards(filename, "SPC")
    spcs = Array{SpcCard,1}(undef,nf)
    i = 0
    f = open(filename, "r")
    for lines in readlines(f)
        line = padline(lines)
        if line[1:3]=="SPC"
            sid  = getid(line[9:16])
            nid  = getid(line[17:24])
            ifix = getids(line[25:32])
            ival = getnum(line[33:40])
            i += 1
            spcs[i] = SpcCard(sid, nid, ifix, ival)
        end
    end
    close(f)
    return spcs
end

function read_spcs(filename, ifix::Set, ndpn=6)
    f = open(filename, "r")
    for lines in readlines(f)
        line = padline(lines)
        if line[1:3]=="SPC"
            nid  = getid(line[17:24])
            ifix = getids(line[25:32])
            ival = getnum(line[33:40])
            for i in ival
                push!(ifix, (nid-1)*ndpn + i)
            end
        end
    end
    close(f)
end

end # of module NastranInput
# ----------------------------------------------------------------------

using .NastranInput: read_mesh, read_forces

filename = "./example.nas";

nodes, elements = NastranInput.read_mesh(filename);
forces = NastranInput.read_forces(filename)
spcs = NastranInput.read_spcs(filename)
