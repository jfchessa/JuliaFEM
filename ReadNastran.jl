module ReadNastran

REALTYPE = Float64
IDTYPE = Int64
FSIZE = 8

function opennasfile(filename)
    """
    f = opennasfile(filename)
    Opens a Nastran file
    """
    f = open(filename, "r")
end

function iscard(cname, line)
    """
    iscard(cname, line)
    Checks to see if the line is a card of type cname
    """
    if cmp(cname,line[1:min(length(cname),length(line))]) == 0
        return true
    end
    return false
end

function numcard(filename, cname)
    """
    n = numcard(filename, cname)
    Returns the number of cards of cname in the Nastran file
    """
    l = length(cname)
    n = 0
    f = opennasfile(filename)
    while !eof(f)
        line = readline(f)
        if iscard(cname, line)
            n += 1
        end
    end
    return n
end

function numgrid(filename)
    """
    n = numgrid(filename)
    Returns the number of GRID cards in the Nastran file
    """
    return numcard(filename, "GRID")
end

function str2float(fstr,v=0.0)
    try
        return parse(REALTYPE, fstr)
    catch # probably something like 12.3+3
        ss1 = replace(strip(fstr),"+"=>"E")
        ss  = replace(ss1,"-"=>"E-")
        if ss[1] == 'E'
            ss = chop(ss,head=1,tail=0)
        end
        try
            return parse(REALTYPE,ss)
        catch
            return v
        end
    end
end

function str2int(istr, v=0)
    try
        return parse(IDTYPE,istr)
    catch
        return v
    end
end

function skipline(f)
    cc = ' '
    while true
        if !eof(f)
            cc = read(f,Char)
            if cc=='\r'
                skip(f,1)  # skip the \n
                break
            end
        else
            return
        end
    end
end

function readfields!(f, fields)
    """
    function readfields!(f, fields)
    Reads field data from the next card in the IO stream f in to fields.  
    fields is a 2D array of characters Array{Char}(undef,fieldsize,numfields)
    Skips comment lines.  If a fields is not read (like a blank line) the first
    field (field[:,1]) is set to "BLANK".  The return value is the number of fields 
    read.
    """
    s, n = size(fields)
    cc = ' '
    i = 0
    for (j,c) in enumerate("BLANK")
        fields[j,1] = c
    end
    while i < length(fields)
        if !eof(f)
            cc = read(f, Char)
            if cc == '\r'
                skip(f,1)  # skip the \n
                break
            elseif cc == '+'
                skipline(f)
                skip(f,s)
            elseif cc == '$'
                skipline(f)
            else
                i += 1
                fields[i] = cc
            end
        else
            return 
        end
    end
    if i==length(fields) # we did not read the whole line so goto the next line
        skipline(f)
    end
    return Int64(floor(i/s))
end

function readgrid!(filename, coord, nid)
    """
    readgrid!(filename, coord, nid)
    Reads the GRID data into coord and nid from the Nastran file filename.
    """
    n = 0
    f = opennasfile(filename)
    while !eof(f)
        line = readline(f)
        if iscard("GRID", line)
            n += 1
            nid[n] = str2int(line[9:16])
            x = str2float(line[25:32])
            y = str2float(line[33:40])
            z = str2float(line[41:48])
            coord[1:3, n] = [x, y, z]
        end
    end
    return n
end

function readcards!(filename, cname, vargs...)
    """
    vargs are triplets in the following format 
         (array, cols, fmtfunc), ..... 
    """
    maxf = 9
    for t in vargs
        maxf = max(maximum(t[2]), maxf)
    end
    fields = Array{Char,2}(undef, FSIZE, maxf)
    f = opennasfile(filename)
    n = 0
    while !eof(f)
        nf = readfields!(f, fields)
        if iscard(cname, String(fields[:,1]))  
            n += 1
            for t in vargs
                s = (ndims(t[1]) == 1 ? 1 : size(t[1],1))             
                for (i, ii) in enumerate(t[2])
                    t[1][s*(n-1)+i] = t[3](String(fields[:,ii]))
                end
            end            
        end
    end
    return n
end

# function readconn(filename, cname, conn, pid, eid)
#
# end

function readchexa!(filename, conn, pid, eid)
    """
    readchexa!(filename, conn, pid, eid)
    Reads in 8 and 20 CHEXA elements
    """
    n = 0
    nne = 8
    fields = Array{Char,2}(undef,FSIZE,25)
    f = opennasfile(filename)
    while !eof(f)
        nfields = readfields!(f, fields)
        nne = nfields - 3
        if iscard("CHEXA", String(fields[:,1]))  
            n += 1
            eid[n] = str2int(String(fields[:,2]))
            pid[n] = str2int(String(fields[:,3]))
            for i = 1:nne
                conn[i,n] = str2int(String(fields[:,i+3]))
            end
        end
    end
    return n
end

function ex1()
    dir = "./"
    filename = dir*"block.bdf"
    #filename = dir*"test.bdf"

    nn = ReadNastran.numgrid(filename)
    coord = Array{Float32}(undef,3,nn)
    nid = Array{Int32}(undef,nn)
    ReadNastran.readgrid!(filename, coord, nid)

    ne = ReadNastran.numcard(filename, "CHEXA")
    conn = Array{Int32}(undef,8,ne)
    eid = Array{Int32}(undef,ne)
    pid = Array{Int32}(undef,ne)
    ReadNastran.readchexa!(filename, conn, pid, eid)

    nspc = ReadNastran.numcard(filename, "SPC")
    spci = Array{Int32}(undef,nspc)
    spcv = Array{Float32}(undef,nspc)
    ReadNastran.readcards!(filename, "SPC", (spci,[3],ReadNastran.str2int), (spcv,[5],ReadNastran.str2float))

    # read in CPENTA connectivity
    ne = ReadNastran.numcard(filename, "CPENTA")
    penta = Array{Int32}(undef,6,ne)
    peid = Array{Int32}(undef,ne)
    ReadNastran.readcards!(filename, "CPENTA", 
            (penta, collect(4:9), ReadNastran.str2int),
            (peid, [2], ReadNastran.str2int))
end

end # module ReadNastran

#using .ReadNastran
#ReadNastran.ex1()