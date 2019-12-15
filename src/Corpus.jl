### Document and Corpus data structures for TopicModelsVB.
### Eric Proffitt
### December 3, 2019

### Document mutable struct.
mutable struct Document
	"terms:   A vector{Int} containing keys for the Corpus lex Dict."
	"counts:  A Vector{Int} denoting the counts of each term in the Document."
	"readers: A Vector{Int} denoting the keys for the Corpus users Dict."
	"ratings: A Vector{Int} denoting the ratings for each reader in the Document."
	"title:   The title of the document (String)."

	terms::Vector{Int}
	counts::Vector{Int}
	readers::Vector{Int}
	ratings::Vector{Int}
	title::String

	function Document(;terms=Int[], counts=ones(length(terms)), readers=Int[], ratings=ones(length(readers)), title="")
		all(ispositive.(terms)) || throw(ArgumentError("All terms must be positive integers."))
		all(ispositive.(counts)) || throw(ArgumentError("All counts must be positive integers."))
		all(ispositive.(readers)) || throw(ArgumentError("All readers must be positive integers."))
		all(ispositive.(ratings)) || throw(ArgumentError("All ratings must be positive integers."))
		isequal(length(terms), length(counts)) || throw(ArgumentError("The terms and counts vectors must have the same length."))

		doc = new(terms, counts, readers, ratings, title)
		return doc
	end
end

### Document outer constructor for non-kwarg terms initialization.
Document(terms) = Document(terms=terms)

### Corpus mutable struct.
mutable struct Corpus
	"docs:  A Vector{Document} containing the documents which belong to the Corpus."
	"lex:   A Dict{Int, String} containing a mapping term Int (key) -> term String (value)."
	"users: A Dict{Int, String} containing a mapping user Int (key) -> user String (value)."

	docs::Vector{Document}
	vocab::Dict{Int, String}
	users::Dict{Int, String}

	function Corpus(;docs=Document[], vocab=[], users=[])
		isa(vocab, Dict) || (vocab = Dict(vkey => term for (vkey, term) in enumerate(vocab)))
		isa(users, Dict) || (users = Dict(ukey => user for (ukey, user) in enumerate(users)))

		all(ispositive.(collect(keys(vocab)))) || throw(ArgumentError("All vocab keys must be positive integers."))
		all(ispositive.(collect(keys(users)))) || throw(ArgumentError("All user keys must be positive integers."))

		corp = new(docs, lex, users)
		return corp
	end
end

Base.show(io::IO, doc::Document) = print(io, "Document with:\n * $(length(doc.terms)) terms\n * $(length(doc.readers)) readers")
Base.length(doc::Document) = length(doc.terms)
Base.size(doc::Document) = sum(doc.counts)
Base.in(doc::Document, corp::Corpus) = in(doc, corp.docs)

function readcorp(;docfile::AbstractString="", lexfile::AbstractString="", userfile::AbstractString="", titlefile::AbstractString="", delim::Char=',', counts::Bool=false, readers::Bool=false, ratings::Bool=false)	
	(ratings <= readers) || (ratings = false; warn("Ratings require readers, ratings switch set to false."))
	(!isempty(docfile) | isempty(titlefile)) || warn("No docfile, titles will not be assigned.")

	corp = Corpus()
	if !isempty(docfile)
		docs = open(docfile)
		doc_kwargs = [:counts, :readers, :ratings]	
		
		for (d, doc_block) in enumerate(partition(readlines(docs), counts + readers + ratings + 1))
			try
				doc_lines = Vector{Float64}[[parse(Float64, p) for p in split(line, delim)] for line in doc_block]			
				doc_input = zip(doc_kwargs[[counts, readers, ratings]], doc_lines[2:end])
				push!(corp, Document(doc_lines[1]; doc_input...))
			catch
				error("Document $d beginning on line $((d - 1) * (counts + readers + ratings) + d) failed to load.")
			end
		end

	else
		warn("No docfile, topic models cannot be trained without documents.")
	end

	if !isempty(lexfile)
		vocab = readdlm(lexfile, '\t', comments=false)
		lkeys = vocab[:,1]
		terms = [string(j) for j in vocab[:,2]]
		corp.vocab = Dict{Int, String}(zip(vkeys, terms))
		@assert all(ispositive.(collect(keys(corp.vocab))))
	end

	if !isempty(userfile)
		users = readdlm(userfile, '\t', comments=false)
		ukeys = users[:,1]
		users = [string(u) for u in users[:,2]]
		corp.users = Dict{Int, String}(zip(ukeys, users))
		@assert all(ispositive.(collect(keys(corp.users))))
	end

	if !isempty(titlefile)
		titles = readdlm(titlefile, '\n', String)
		for (d, doc) in enumerate(corp)
			doc.title = titles[d]
		end
	end

	return corp
end

function writecorp(corp::Corpus; docfile::AbstractString="", vocabfile::AbstractString="", userfile::AbstractString="", titlefile::AbstractString="", delim::Char=',', counts::Bool=false, readers::Bool=false, ratings::Bool=false)	
	(ratings <= readers) || (ratings = false; warn("Ratings require readers, ratings switch set to false."))

	if !isempty(docfile)
		doc_kwargs = [:counts, :readers, :ratings]
		doc_kwargs = doc_kwargs[[counts, readers, ratings]]
		docfile = open(docfile, "w")
		for doc in corp
			write(docfile, join(doc.terms, delim), '\n')
			for arg in doc_kwargs
				write(docfile, join(doc.(arg), delim), '\n')
			end
		end
		close(docfile)
	end

	if !isempty(vocabfile)
		vkeys = sort(collect(keys(corp.vocab)))
		vocab = zip(vkeys, [corp.vocab[vkey] for vkey in vkeys])
		writedlm(vocabfile, vocab)
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

### The _corp and _docs functions are designed to provide safe methods for modifying corpora.
### The _corp functions only meaningfully modify the Corpus object.
### In so far as the _corp functions modify Document objects, it only amounts to a possible relabeling of the keys in the documents.
### The _doc functions only modify the Document objects attached the corpus, they do not modify the Corpus object.
### The exception to the above rule is the stop_corp! function, which removes stop words from both the Corpus vocab dictionary and associated keys in the documents.

function stop_corp!(corp::Corpus)
	"Filter stop words in the associated corpus."

	version = "v$(VERSION.major).$(VERSION.minor)"	

	stopwords = vec(readdlm(pwd() * "/.julia/$version/topicmodelsvb/datasets/stopwords.txt", String))
	stop_keys = filter(vkey -> lowercase(corp.vocab[vkey]) in stopwords, collect(keys(corp.vocab)))
	
	for doc in unique(corp)
		keep = Bool[!(j in stop_keys) for j in doc.terms]
		doc.terms = doc.terms[keep]
		doc.counts = doc.counts[keep]
	end
	nothing
end

function alphabetize_corp!(corp::Corpus; vocab::Bool=true, users::Bool=true)
	"Alphabetize vocab and/or user dictionaries."

	if vocab
		vkeys = sort(collect(keys(corp.vocab)))
		terms = sort(collect(values(corp.vocab)))

		vkey_map = Dict(vkey_old => vkey_new for (vkey_old, vkey_new) in zip(vkeys, vkeys[sortperm(sortperm([corp.vocab[vkey] for vkey in vkeys]))]))
		corp.vocab = Dict(vkey => term for (vkey, term) in zip(vkeys, terms))

		for doc in unique(corp)
			doc.terms = [vkey_map[vkey] for vkey in doc.terms]
		end
	end

	if users
		ukeys = sort(collect(keys(corp.users)))
		users = sort(collect(values(corp.users))))

		ukey_map = Dict(ukey_old => ukey_new for (ukey_old, ukey_new) in zip(ukeys, ukeys[sortperm(sortperm([corp.users[ukey] for ukey in ukeys]))]))
		corp.users = Dict(ukey => user for (ukey, user) in zip(ukeys, user))

		for doc in unique(corp)
			doc.users = [ukey_map[ukey] for ukey in doc.users]
		end
	end
	nothing
end

function abridge_corp!(corp::Corpus, n::Integer)
	"All terms which appear less than or equal to n times in the corpus are removed from all documents."

	doc_vkeys = Set(vcat([doc.terms for doc in unique(corp)]...))
	vocab_count = Dict(Int(j) => 0 for j in doc_vkeys)
	
	for doc in unique(corp), (j, c) in zip(doc.terms, doc.counts)
		vocab_count[j] += c
	end

	for doc in unique(corp)
		keep = Bool[vocab_count[j] > n for j in doc.terms]
		doc.terms = doc.terms[keep]
		doc.counts = doc.counts[keep]
	end
	nothing
end

function compact_corp!(corp::Corpus; vocab::Bool=true, users::Bool=true)
	"Relabel vocab and/or user keys so that they form a unit range."

	if vocab
		vkeys = sort(collect(keys(corp.vocab)))
		vkey_map = Dict(vkey => v for (v, vkey) in enumerate(vkeys))
		corp.vocab = Dict(vkey_map[vkey] => corp.vocab[vkey] for vkey in keys(corp.vocab))

		for doc in unique(corp)
			doc.terms = [vkey_map[vkey] for vkey in doc.terms]
		end
	end

	if users
		ukeys = sort(collect(keys(corp.users)))
		ukey_map = Dict(ukey => u for (u, ukey) in enumerate(ukeys))
		corp.users = Dict(ukey_map[ukey] => corp.users[ukey] for ukey in keys(corp.users))

		for doc in unique(corp)
			doc.users = [ukey_map[ukey] for ukey in doc.users]
		end
	end
	nothing
end

function trim_corp!(corp::Corpus; vocab::Bool=true, users::Bool=true)
	"Those keys which appear in the corpus vocab and/or user dictionaries but not in any of the documents are removed from the corpus."

	if vocab
		doc_vkeys = Set(vcat([doc.terms for doc in corp]...))
		corp.vocab = Dict(vkey => corp.vocab[vkey] for vkey in intersect(keys(corp.vocab), doc_vkeys))
	end

	if users
		doc_ukeys = Set(vcat([doc.terms for doc in corp]...))
		corp.users = Dict(ukey => corp.users[ukey] for ukey in intersect(keys(corp.users), doc_ukeys))
	end
	nothing
end

function trim_docs!(corp::Corpus; terms::Bool=true, readers::Bool=true)
	"Those vocab and/or user keys which appear in documents but not in the corpus dictionaries are removed from the documents."

	if terms
		doc_vkeys = Set(vcat([doc.terms for doc in corp]...))
		bogus_vkeys = setdiff(doc_vkeys, keys(corp.vocab))
		for doc in corp
			keep = Bool[!(j in bogus_vkeys) for j in doc.terms]
			doc.terms = doc.terms[keep]
			doc.counts = doc.counts[keep]
		end
	end

	if terms
		doc_ukeys = Set(vcat([doc.readers for doc in corp]...))
		bogus_ukeys = setdiff(doc_ukeys, keys(corp.users))
		for doc in corp
			keep = Bool[!(u in bogus_ukeys) for u in doc.readers]
			doc.readers = doc.readers[keep]
			doc.ratings = doc.ratings[keep]
		end
	end
	nothing
end

function condense_docs!(corp::Corpus)
	"Ignore term order in documents."
	"Multiple seperate occurrences of terms are stacked and their associated counts increased."

	for doc in unique(corp)
		docdict = Dict(j => 0 for j in doc.terms)
		for (j, c) in zip(doc.terms, doc.counts)
			docdict[j] += c
		end

		doc.terms = collect(keys(docdict))
		doc.counts = collect(values(docdict))
	end
	nothing
end

function remove_empty_docs!(corp::Corpus)
	"Documents with no terms are removed from the corpus."

	keep = Bool[length(doc.terms) > 0 for doc in corp]
	corp.docs = corp[keep]
	nothing
end

function pad_corp!(corp::Corpus; vocab::Bool=true, users::Bool=true)
	"Enter generic values for vocab and/or user keys which appear in documents but not in the vocab/user dictionaries."

	if vocab
		doc_vkeys = Set(vcat([doc.terms for doc in corp]...))
		for vkey in setdiff(doc_vkeys, keys(corp.vocab))
			corp.vocab[vkey] = string(join(["#term", vkey]))
		end
	end

	if users
		doc_ukeys = Set(vcat([doc.readers for doc in corp]...))
		for ukey in setdiff(doc_ukeys, keys(corp.users))
			corp.users[ukey] = string(join(["#user", ukey]))
		end
	end
	nothing
end

function fix_corp!(corp::Corpus, pad::Bool=false)
	"Generic function to ensure that a Corpus object can be loaded ino a TopicModel object."

	if pad
		pad_corpus!(corp)
	else
		trim_docs!(corp)
	end

	compact_corp!(corp)
	nothing
end

Base.show(io::IO, corp::Corpus) = print(io, "Corpus with:\n * $(length(corp)) docs\n * $(length(corp.lex)) lex\n * $(length(corp.users)) users")
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



function showdocs{T<:Integer}(corp::Corpus, ds::Vector{<:Integer})
	@assert checkbounds(Bool, 1:length(corp), ds) "Some document indices outside docs range."
	
	for d in ds
		doc = corp[d]
		@juliadots "doc $d\n"
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

showdocs(corp::Corpus, ds::UnitRange{<:Integer}) = showdocs(corp, collect(ds))
showdocs(corp::Corpus, d::Integer) = showdocs(corp, [d])
showdocs(corp::Corpus, doc::Document) = showdocs(corp, [doc])

getvocab(corp::Corpus) = sort(collect(values(corp.vocab)))
getusers(corp::Corpus) = sort(collect(values(corp.users)))



######################################
### Pre-packaged Dataset Shortcuts ###
######################################

function readcorp(corp_symbol::Symbol)
	version = "v$(VERSION.major).$(VERSION.minor)"

	if corp_symbol == :nsf
		docfile = homedir() * "/.julia/$version/topicmodelsvb/datasets/nsf/nsfdocs.txt"
		lexfile = homedir() * "/.julia/$version/topicmodelsvb/datasets/nsf/nsflex.txt"
		titlefile = homedir() * "/.julia/$version/topicmodelsvb/datasets/nsf/nsftitles.txt"
		corp = readcorp(docfile=docfile, lexfile=lexfile, titlefile=titlefile, counts=true, stamps=true)

	elseif corp_symbol == :citeu
		docfile = homedir() * "/.julia/$version/topicmodelsvb/datasets/citeu/citeudocs.txt"
		vocabfile = homedir() * "/.julia/$version/topicmodelsvb/datasets/citeu/citeulex.txt"
		userfile = homedir() * "/.julia/$version/topicmodelsvb/datasets/citeu/citeuusers.txt"
		titlefile = homedir() * "/.julia/$version/topicmodelsvb/datasets/citeu/citeutitles.txt"
		corp = readcorp(docfile=docfile, lexfile=lexfile, userfile=userfile, titlefile=titlefile, counts=true, readers=true)
		padcorp!(corp)

	else
		println("Included corpora:\n:nsf\n:citeu")
		corp = nothing
	end

	return corp
end
