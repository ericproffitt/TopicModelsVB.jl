#################
#				#
# Document Type #
#				#
#################

type Document
	terms::Vector{Int}
	counts::Vector{Int}
	readers::Vector{Int}
	ratings::Vector{Int}
	stamp::Float64
	title::UTF8String

	function Document(terms; counts=ones(length(terms)), readers=Int[], ratings=ones(length(readers)), stamp=-Inf, title="")
		doc = new(terms, counts, readers, ratings, stamp[1], title)
		checkdoc(doc)
		return doc
	end
end

function Base.show(io::IO, doc::Document)
	print(io, "Document with:\n * $(length(doc.terms)) terms\n * $(length(doc.readers)) readers") 
	if isfinite(doc.stamp)
		print(io, "\n * $(doc.stamp) stamp")
	end
end

Base.length(doc::Document) = length(doc.terms)
Base.size(doc::Document) = sum(doc.counts)

function checkdoc(doc::Document)
	pass =
	(!isempty(doc.terms)
	& all(ispositive(doc.terms))
	& all(ispositive(doc.counts))
	& isequal(length(doc.terms), length(doc.counts))
	& all(ispositive(doc.readers))
	& all(ispositive(doc.ratings))
	& isequal(length(doc.readers), length(doc.ratings)))
	return pass	
end



###############
#			  #
# Corpus Type #
#			  #
###############

type Corpus
	docs::Vector{Document}
	lex::Dict{Int, UTF8String}
	users::Dict{Int, UTF8String}

	function Corpus(;docs=Document[], lex=[], users=[])
		isa(lex, Dict) || (lex = [lkey => term for (lkey, term) in enumerate(lex)])
		isa(users, Dict) || (users = [ukey => user for (ukey, user) in enumerate(users)])

		corp = new(docs, lex, users)
		checkcorp(corp)
		return corp
	end
end

Base.show(io::IO, corp::Corpus) = print(io, "Corpus with:\n * $(length(corp)) docs\n * $(length(corp.lex)) lex\n * $(length(corp.users)) users")
Base.in(doc::Document, corp::Corpus) = in(doc, corp.docs)
Base.start(corp::Corpus) = 1
Base.next(corp::Corpus, d::Int) = corp.docs[d], d + 1
Base.done(corp::Corpus, d::Int) = length(corp.docs) == d - 1
Base.push!(corp::Corpus, doc::Document) = push!(corp.docs, doc)
Base.pop!(corp::Corpus) = pop!(corp.docs)
Base.unshift!(corp::Corpus, doc::Document) = unshift!(corp.docs, doc)
Base.unshift!(corp::Corpus, docs::Vector{Document}) = unshift!(corp.docs, docs)
Base.shift!(corp::Corpus) = shift!(corp.docs)
Base.insert!(corp::Corpus, d::Int, doc::Document) = insert!(corp.docs, d, doc)
Base.deleteat!(corp::Corpus, d::Int) = deleteat!(corp.docs, d)
Base.deleteat!(corp::Corpus, ds::Vector{Int}) = deleteat!(corp.docs, ds)
Base.deleteat!(corp::Corpus, ds::UnitRange{Int}) = deleteat!(corp.docs, ds)
Base.getindex(corp::Corpus, d::Int) = getindex(corp.docs, d)
Base.getindex(corp::Corpus, ds::Vector{Int}) = getindex(corp.docs, ds)
Base.getindex(corp::Corpus, ds::UnitRange{Int}) = getindex(corp.docs, ds)
Base.getindex(corp::Corpus, ds::Vector{Bool}) = getindex(corp.docs, find(ds))
Base.setindex!(corp::Corpus, doc::Document, d::Int) = setindex!(corp.docs, doc, d)
Base.setindex!(corp::Corpus, docs::Vector{Document}, ds::Vector{Int}) = setindex!(corp.docs, docs, ds)
Base.setindex!(corp::Corpus, docs::Vector{Document}, ds::UnitRange{Int}) = setindex!(corp.docs, docs, ds)
Base.findfirst(corp::Corpus, doc::Document) = findfirst(corp.docs, doc)
Base.findin(corp::Corpus, docs::Vector{Document}) = findin(corp.docs, docs)
Base.findin(corp::Corpus, doc::Document) = findin(corp, [doc])
Base.length(corp::Corpus) = length(corp.docs)
Base.size(corp::Corpus) = (length(corp), length(corp.lex), length(corp.users))
Base.copy(corp::Corpus) = Corpus(docs=copy(corp.docs), lex=copy(corp.lex), users=copy(corp.users))
Base.endof(corp::Corpus) = length(corp)

function checkcorp(corp::Corpus)
	pass = true
	for (d, doc) in enumerate(corp)
		checkdoc(doc) || (println("Document $d failed check."); pass=false)
	end
	@assert all(ispositive(collect(keys(corp.lex))))
	@assert all(ispositive(collect(keys(corp.users))))
	return pass
end



#############################################
#											#
# Functions for Reading and Writing Corpora #
#											#
#############################################

function readcorp(;docfile::AbstractString="", lexfile::AbstractString="", userfile::AbstractString="", titlefile::AbstractString="", delim::Char=',', counts::Bool=false, readers::Bool=false, ratings::Bool=false, stamps::Bool=false)	
	(ratings <= readers) || (ratings = false; warn("Ratings require readers, ratings switch set to false."))
	(!isempty(docfile) | isempty(titlefile)) || warn("No docfile, titles will not be assigned.")
	stamp = stamps

	corp = Corpus()
	if !isempty(docfile)
		docs = open(docfile)
		dockwargs = [:counts, :readers, :ratings, :stamp]	
		for (d, docblock) in enumerate(partition(readlines(docs), counts + readers + ratings + stamp + 1))
			try
			doclines = Vector{Float64}[[parse(Float64, p) for p in split(line, delim)] for line in docblock]			
			docinput = zip(dockwargs[[counts, readers, ratings, stamp]], doclines[2:end])
			push!(corp, Document(doclines[1]; docinput...))
			catch error("Document $d beginning on line $((d - 1) * (counts + readers + ratings + stamp) + d) failed to load.")
			end
		end
	else warn("No docfile, topic models cannot be trained without documents.")
	end

	if !isempty(lexfile)
		lex = readdlm(lexfile, '\t', comments=false)
		lkeys = lex[:,1]
		terms = [string(j) for j in lex[:,2]]
		corp.lex = Dict{Int, UTF8String}(zip(lkeys, terms))
		@assert all(ispositive(collect(keys(corp.lex))))
	end

	if !isempty(userfile)
		users = readdlm(userfile, '\t', comments=false)
		ukeys = users[:,1]
		users = [string(u) for u in users[:,2]]
		corp.users = Dict{Int, UTF8String}(zip(ukeys, users))
		@assert all(ispositive(collect(keys(corp.users))))
	end

	if !isempty(titlefile)
		titles = readdlm(titlefile, '\t', UTF8String)
		for (d, doc) in enumerate(corp)
			doc.title = titles[d]
		end
	end

	return corp
end

function writecorp(corp::Corpus; docfile::AbstractString="", lexfile::AbstractString="", userfile::AbstractString="", titlefile::AbstractString="", delim::Char=',', counts::Bool=false, readers::Bool=false, ratings::Bool=false, stamps::Bool=false)	
	(ratings <= readers) || (ratings = false; warn("Ratings require readers, ratings switch set to false."))
	stamp = stamps

	if !isempty(docfile)
		dockwargs = [:counts, :readers, :ratings, :stamp]
		dockwargs = dockwargs[[counts, readers, ratings, stamp]]
		docfile = open(docfile, "w")
		for doc in corp
			write(docfile, join(doc.terms, delim), '\n')
			for arg in dockwargs
				write(docfile, join(doc.(arg), delim), '\n')
			end
		end
		close(docfile)
	end

	if !isempty(lexfile)
		lkeys = sort(collect(keys(corp.lex)))
		lex = zip(lkeys, [corp.lex[lkey] for lkey in lkeys])
		writedlm(lexfile, lex)
	end

	if !isempty(userfile)
		ukeys = sort(collect(keys(corp.users)))
		users = zip(ukeys, [corp.users[ukey] for ukey in ukeys])
		writedlm(userfile, users)
	end

	if !isempty(titlefile)
		writedlm(titlefile, [doc.title for doc in corp])
	end
	nothing
end



###################
#				  #
# corp! Functions #
#				  #
###################

function abridgecorp!(corp::Corpus; stop::Bool=false, order::Bool=true, abr::Integer=1)
	if stop
		stopwords = vec(readdlm(pwd() * "/.julia/v0.4/topicmodelsvb/datasets/stopwords.txt", UTF8String))
		stopkeys = filter(j -> lowercase(corp.lex[j]) in stopwords, collect(keys(corp.lex)))
		for doc in corp
			keep = Bool[!(j in stopkeys) for j in doc.terms]
			doc.terms = doc.terms[keep]
			doc.counts = doc.counts[keep]
		end
	end

	if !order
		for doc in corp
			docdict = [Int(j) => 0 for j in doc.terms]
			for (j, c) in zip(doc.terms, doc.counts)
				docdict[j] += c
			end
			doc.terms = collect(keys(docdict))
			doc.counts = collect(values(docdict))
		end
	end

	if abr > 1
		doclkeys = Set(vcat([doc.terms for doc in corp]...))
		lexcount = [Int(j) => 0 for j in doclkeys]
		for doc in corp, (j, c) in zip(doc.terms, doc.counts)
			lexcount[j] += c
		end

		for doc in corp
			keep = Bool[lexcount[j] >= abr for j in doc.terms]
			doc.terms = doc.terms[keep]
			doc.counts = doc.counts[keep]
		end

		keep = Bool[length(doc.terms) > 0 for doc in corp]
		corp.docs = corp[keep]
	end
	nothing
end

function trimcorp!(corp::Corpus; lex::Bool=true, terms::Bool=true, users::Bool=true, readers::Bool=true)
	if lex		
		doclkeys = Set(vcat([doc.terms for doc in corp]...))
		corp.lex = [lkey => corp.lex[lkey] for lkey in intersect(keys(corp.lex), doclkeys)]
	end

	if terms
		doclkeys = Set(vcat([doc.terms for doc in corp]...))
		boguslkeys = setdiff(doclkeys, keys(corp.lex))
		for doc in corp
			keep = Bool[!(j in boguslkeys) for j in doc.terms]
			doc.terms = doc.terms[keep]
			doc.counts = doc.counts[keep]
		end
	end
	
	if users
		docukeys = Set(vcat([doc.readers for doc in corp]...))
		corp.users = [ukey => corp.users[ukey] for ukey in intersect(keys(corp.users), docukeys)]
	end

	if readers
		docukeys = Set(vcat([doc.readers for doc in corp]...))	
		bogusukeys = setdiff(docukeys, keys(corp.users))
		for doc in corp
			keep = Bool[!(u in bogusukeys) for u in doc.readers]
			doc.readers = doc.readers[keep]
			doc.ratings = doc.ratings[keep]
		end
	end
	nothing
end

function compactcorp!(corp::Corpus; lex::Bool=true, users::Bool=true, alphabetize::Bool=true)	
	if lex
		lkeys = sort(collect(keys(corp.lex)))
		lkeymap = zip(lkeys, 1:length(corp.lex))
		if alphabetize
			alphabetdict = [j => lkey for (j, lkey) in zip(sortperm([corp.lex[lkey] for lkey in lkeys]), 1:length(corp.lex))]
			lkeydict = [lkey => alphabetdict[j] for (lkey, j) in lkeymap]	
		else
			lkeydict = [lkey => j for (lkey, j) in lkeymap]
		end
		
		corp.lex = [lkeydict[lkey] => corp.lex[lkey] for lkey in keys(corp.lex)]	
		for lkey in keys(corp.lex)
			try if corp.lex[lkey][1:5] == "#term"; corp.lex[lkey] = "#term$(lkey)"; end
			end
		end
	end

	if users
		ukeys = sort(collect(keys(corp.users)))
		ukeymap = zip(ukeys, 1:length(corp.users))
		if alphabetize
			alphabetdict = [r => ukey for (r, ukey) in zip(sortperm([corp.users[ukey] for ukey in ukeys]), 1:length(corp.users))]
			ukeydict = [ukey => alphabetdict[r] for (ukey, r) in ukeymap]
		else
			ukeydict = [ukey => r for (ukey, r) in ukeymap]
		end

		corp.users = [ukeydict[ukey] => corp.users[ukey] for ukey in keys(corp.users)]
		for ukey in keys(corp.users)
			try if corp.users[ukey][1:5] == "#user"; corp.users[ukey] = "#user$(ukey)"; end
			end
		end
	end

	for doc in corp
		if lex; doc.terms = [lkeydict[lkey] for lkey in doc.terms]; end
		if users; doc.readers = [ukeydict[ukey] for ukey in doc.readers]; end
	end
	nothing
end

function padcorp!(corp::Corpus; lex::Bool=true, users::Bool=true)
	if lex
		doclkeys = Set(vcat([doc.terms for doc in corp]...))
		for lkey in setdiff(doclkeys, keys(corp.lex))
			corp.lex[lkey] = string(join(["#term",lkey]))
		end
	end
	if users
		docukeys = Set(vcat([doc.readers for doc in corp]...))
		for ukey in setdiff(docukeys, keys(corp.users))
			corp.users[ukey] = string(join(["#user",ukey]))
		end
	end
	nothing
end

function cullcorp!(corp::Corpus; lex::Bool=false, users::Bool=false, len::Integer=1)
	lexkeys = keys(corp.lex)
	userkeys = keys(corp.users)
	bogusdocs = Int[]
	for (d, doc) in enumerate(corp)
		if lex
			for j in doc.terms
				if !(j in lexkeys)
					push!(bogusdocs, d)
					break
				end
			end
		end

		if users & !(d in bogusdocs)
			for u in doc.readers
				if !(u in userkeys)
					push!(bogusdocs, d)
					break
				end
			end
		end

		if (len > 1) & !(d in bogusdocs)
			if length(doc) < len
				push!(bogusdocs, d)
			end
		end

		if isempty(doc.terms) & !(d in bogusdocs)
			push!(bogusdocs, d)
		end
	end

	deleteat!(corp, bogusdocs)
	nothing
end

function fixcorp!(corp::Corpus; lex::Bool=true, terms::Bool=true, users::Bool=true, readers::Bool=true, stop::Bool=false, order::Bool=true, abr::Integer=1, len::Integer=1, alphabetize::Bool=true)
	println("Abridging corpus..."); abridgecorp!(corp, stop=stop, order=order, abr=abr)
	println("Trimming corpus..."); trimcorp!(corp, lex=lex, terms=terms, users=users, readers=readers)
	println("Culling corpus..."); cullcorp!(corp, len=len)	
	println("Compacting corpus..."); compactcorp!(corp, lex=lex, users=users, alphabetize=alphabetize)
	nothing
end



#########################################
#										#
# Document and Corpus Display Functions #
#										#
#########################################

function showdocs{T<:Integer}(corp::Corpus, ds::Vector{T})
	@assert checkbounds(Bool, length(corp), ds) "Some document indices outside docs range."
	
	for d in ds
		doc = corp[d]
		@juliadots "Doc: $d\n"
		if !isempty(doc.title)
			@juliadots "$(doc.title)\n"
		end
		println(join([corp.lex[lkey] for lkey in corp[d].terms], " "), '\n')
	end
end

function showdocs(corp::Corpus, docs::Vector{Document})
	ds = findin(corp, docs)
	showdocs(corp, ds)
end

showdocs{T<:Integer}(corp::Corpus, ds::UnitRange{T}) = showdocs(corp, collect(ds))
showdocs(corp::Corpus, d::Integer) = showdocs(corp, [d])
showdocs(corp::Corpus, doc::Document) = showdocs(corp, [doc])

getlex(corp::Corpus) = sort(collect(values(corp.lex)))
getusers(corp::Corpus) = sort(collect(values(corp.users)))



##################################
#								 #
# Pre-packaged Dataset Shortcuts # 
#								 #
##################################

function readcorp(corpsym::Symbol)
	v = "v$(VERSION.major).$(VERSION.minor)"

	if corpsym == :nsf
		docfile = homedir() * "/.julia/$v/topicmodelsvb/datasets/nsf/nsfdocs.txt"
		lexfile = homedir() * "/.julia/$v/topicmodelsvb/datasets/nsf/nsflex.txt"
		titlefile = homedir() * "/.julia/$v/topicmodelsvb/datasets/nsf/nsftitles.txt"
		corp = readcorp(docfile=docfile, lexfile=lexfile, titlefile=titlefile, counts=true, stamps=true)

	elseif corpsym == :citeu
		docfile = homedir() * "/.julia/$v/topicmodelsvb/datasets/citeu/citeudocs.txt"
		lexfile = homedir() * "/.julia/$v/topicmodelsvb/datasets/citeu/citeulex.txt"
		userfile = homedir() * "/.julia/$v/topicmodelsvb/datasets/citeu/citeuusers.txt"
		titlefile = homedir() * "/.julia/$v/topicmodelsvb/datasets/citeu/citeutitles.txt"
		corp = readcorp(docfile=docfile, lexfile=lexfile, userfile=userfile, titlefile=titlefile, counts=true, readers=true)
		padcorp!(corp)

	elseif corpsym == :mac
		docfile = homedir() * "/.julia/$v/topicmodelsvb/datasets/mac/macdocs.txt"
		lexfile = homedir() * "/.julia/$v/topicmodelsvb/datasets/mac/maclex.txt"
		titlefile = homedir() * "/.julia/$v/topicmodelsvb/datasets/mac/mactitles.txt"
		corp = readcorp(docfile=docfile, lexfile=lexfile, titlefile=titlefile, counts=true, stamps=true)

	else
		println("Included corpora:\n:nsf\n:citeu\n:mac")
		corp = nothing
	end

	return corp
end


