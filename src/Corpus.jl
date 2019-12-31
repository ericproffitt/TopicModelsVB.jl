### Document and Corpus data structures for TopicModelsVB.
### Eric Proffitt
### December 3, 2019

mutable struct Document
	"Document mutable struct"

	"terms:   A vector{Int} containing keys for the Corpus vocab Dict."
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
		doc = new(terms, counts, readers, ratings, title)
		check_doc(doc)
		return doc
	end
end

### Document outer constructors.
Document(terms) = Document(terms=terms)

struct DocumentError <: Exception
    msg::String
end

Base.showerror(io::IO, e::DocumentError) = print(io, "DocumentError: ", e.msg)

function check_doc(doc::Document)
	"Check Document parameters."

	all(doc.terms .> 0)									|| throw(DocumentError("All terms must be positive integers."))
	all(doc.counts .> 0)								|| throw(DocumentError("All counts must be positive integers."))
	isequal(length(doc.terms), length(doc.counts))		|| throw(DocumentError("The terms and counts vectors must have the same length."))
	all(doc.readers .> 0)								|| throw(DocumentError("All readers must be positive integers."))
	all(doc.ratings .> 0)								|| throw(DocumentError("All ratings must be positive integers."))
	isequal(length(doc.readers), length(doc.ratings))	|| throw(DocumentError("The readers and ratings vectors must have the same length."))
 	nothing
 end

mutable struct Corpus
	"Corpus mutable struct."

	"docs:  A Vector{Document} containing the documents which belong to the Corpus."
	"vocab: A Dict{Int, String} containing a mapping term Int (key) => term String (value)."
	"users: A Dict{Int, String} containing a mapping user Int (key) => user String (value)."

	docs::Vector{Document}
	vocab::Dict{Int, String}
	users::Dict{Int, String}

	function Corpus(;docs=Document[], vocab=[], users=[])
		isa(vocab, Dict) || (vocab = Dict(vkey => term for (vkey, term) in enumerate(vocab)))
		isa(users, Dict) || (users = Dict(ukey => user for (ukey, user) in enumerate(users)))

		corp = new(docs, vocab, users)
		check_corp(corp)
		return corp
	end
end

### Corpus outer constructors.
Corpus(docs::Vector{Document}) = Corpus(docs=docs)
Corpus(docs::Vector{Document}; vocab=[], users=[]) = Corpus(docs=docs, vocab=vocab, users=users)
Corpus(doc::Document) = Corpus(docs=[doc])
Corpus(doc::Document; vocab=[], users=[]) = Corpus(docs=[doc], vocab=vocab, users=users)

struct CorpusError <: Exception
    msg::String
end

Base.showerror(io::IO, e::CorpusError) = print(io, "CorpusError: ", e.msg)

function check_corp(corp::Corpus)
	"Check Corpus parameters."

	for (d, doc) in enumerate(corp)
		try
			check_doc(doc)
		catch
			println("Document $d failed check.")
			check_doc(doc)
		end
	end

	all(collect(keys(corp.vocab)) .> 0) || throw(CorpusError("All vocab keys must be positive integers."))
	all(collect(keys(corp.users)) .> 0) || throw(CorpusError("All user keys must be positive integers."))
	nothing
end

Base.show(io::IO, doc::Document) = print(io, "Document with:\n * $(length(doc.terms)) terms\n * $(length(doc.readers)) readers")
Base.length(doc::Document) = length(doc.terms)
Base.size(doc::Document) = sum(doc.counts)
Base.in(doc::Document, corp::Corpus) = in(doc, corp.docs)
Base.isempty(doc::Document) = length(doc) == 0

Base.show(io::IO, corp::Corpus) = print(io, "Corpus with:\n * $(length(corp)) docs\n * $(length(corp.vocab)) vocab\n * $(length(corp.users)) users")
Base.iterate(corp::Corpus, d=1) = Base.iterate(corp.docs, d)
Base.push!(corp::Corpus, doc::Document) = push!(corp.docs, doc)
Base.pop!(corp::Corpus) = pop!(corp.docs)
Base.pushfirst!(corp::Corpus, doc::Document) = pushfirst!(corp.docs, doc)
Base.pushfirst!(corp::Corpus, docs::Vector{Document}) = pushfirst!(corp.docs, docs)
Base.popfirst!(corp::Corpus) = popfirst!(corp.docs)
Base.insert!(corp::Corpus, d::Int, doc::Document) = insert!(corp.docs, d, doc)
Base.deleteat!(corp::Corpus, d::Int) = deleteat!(corp.docs, d)
Base.deleteat!(corp::Corpus, doc_indices::Vector{Int}) = deleteat!(corp.docs, doc_indices)
Base.deleteat!(corp::Corpus, doc_indices::UnitRange{Int}) = deleteat!(corp.docs, doc_indices)
Base.getindex(corp::Corpus, d::Int) = getindex(corp.docs, d)
Base.getindex(corp::Corpus, doc_indices::Vector{Int}) = getindex(corp.docs, doc_indices)
Base.getindex(corp::Corpus, doc_indices::UnitRange{Int}) = getindex(corp.docs, doc_indices)
Base.getindex(corp::Corpus, doc_bool::Vector{Bool}) = getindex(corp.docs, doc_bool)
Base.setindex!(corp::Corpus, doc::Document, d::Int) = setindex!(corp.docs, doc, d)
Base.setindex!(corp::Corpus, docs::Vector{Document}, doc_indices::Vector{Int}) = setindex!(corp.docs, docs, doc_indices)
Base.setindex!(corp::Corpus, docs::Vector{Document}, doc_indices::UnitRange{Int}) = setindex!(corp.docs, docs, doc_indices)
Base.findfirst(corp::Corpus, doc::Document) = findfirst(corp.docs, doc)
Base.findall(corp::Corpus, docs::Vector{Document}) = findall((in)corp.docs, docs)
Base.findall(corp::Corpus, doc::Document) = findall(corp, [doc])
Base.length(corp::Corpus) = length(corp.docs)
Base.size(corp::Corpus) = (length(corp), length(corp.vocab), length(corp.users))
Base.copy(corp::Corpus) = Corpus(docs=copy(corp.docs), vocab=copy(corp.vocab), users=copy(corp.users))
Base.lastindex(corp::Corpus) = length(corp)
Base.enumerate(corp::Corpus) = enumerate(corp.docs)
Base.unique(corp::Corpus) = Corpus(docs=unique(corp.docs), vocab=corp.vocab, users=corp.users)

function showdocs(corp::Corpus, doc_indices::Vector{<:Integer})
	"Display document(s) in readable format."

	@assert checkbounds(Bool, 1:length(corp), doc_indices) "Some document indices outside docs range."
	
	for d in doc_indices
		doc = corp[d]
		@juliadots "Document $d\n"
		if !isempty(doc.title)
			@juliadots "$(doc.title)\n"
		end
		println(Crayon(bold=false), join([corp.vocab[vkey] for vkey in corp[d].terms], " "), '\n')
	end
end

function showdocs(corp::Corpus, docs::Vector{Document})
	doc_indices = findall(corp, docs)
	showdocs(corp, doc_indices)
end

showdocs(corp::Corpus, doc_range::UnitRange{<:Integer}) = showdocs(corp, collect(doc_range))
showdocs(corp::Corpus, d::Integer) = showdocs(corp, [d])
showdocs(corp::Corpus, doc::Document) = showdocs(corp, [doc])

getvocab(corp::Corpus) = sort(collect(values(corp.vocab)))
getusers(corp::Corpus) = sort(collect(values(corp.users)))

function readcorp(;docfile::String="", vocabfile::String="", userfile::String="", titlefile::String="", delim::Char=',', counts::Bool=false, readers::Bool=false, ratings::Bool=false)	
	"Load a Corpus object from text file(s)."

	(ratings <= readers) || (ratings = false; warn("Ratings require readers, ratings switch set to false."))
	(!isempty(docfile) | isempty(titlefile)) || warn("No docfile, titles will not be assigned.")

	corp = Corpus()
	if !isempty(docfile)
		docs = open(docfile)
		doc_kwargs = [:terms, :counts, :readers, :ratings]	
		
		for (d, doc_block) in enumerate(Iterators.partition(readlines(docs), counts + readers + ratings + 1))
			try 
				doc_lines = Vector{Int}[[parse(Int, p) for p in split(line, delim)] for line in doc_block]			
				doc_input = zip(doc_kwargs[[true, counts, readers, ratings]], doc_lines)
				push!(corp, Document(;doc_input...))
			
			catch
				error("Document $d beginning on line $((d - 1) * (counts + readers + ratings) + d) failed to load.")
			end
		end

	else
		warn("No docfile, topic models cannot be trained without documents.")
	end

	if !isempty(vocabfile)
		vocab = readdlm(vocabfile, '\t', comments=false)
		vkeys = vocab[:,1]
		terms = [string(j) for j in vocab[:,2]]
		corp.vocab = Dict{Int, String}(zip(vkeys, terms))
		@assert all(collect(keys(corp.vocab)) .> 0)
	end

	if !isempty(userfile)
		users = readdlm(userfile, '\t', comments=false)
		ukeys = users[:,1]
		users = [string(u) for u in users[:,2]]
		corp.users = Dict{Int, String}(zip(ukeys, users))
		@assert all(collect(keys(corp.users)) .> 0)
	end

	if !isempty(titlefile)
		titles = readdlm(titlefile, '\n', String)
		for (d, doc) in enumerate(corp)
			doc.title = titles[d]
		end
	end

	return corp
end

function readcorp(corp_symbol::Symbol)
	"Shortcuts for prepackaged corpora."

	version = "v$(VERSION.major).$(VERSION.minor)"

	if corp_symbol == :nsf
		docfile = homedir() * "/GitHub/TopicModelsVB.jl/datasets/nsf/nsfdocs.txt"
		vocabfile = homedir() * "/GitHub/TopicModelsVB.jl/datasets/nsf/nsfvocab.txt"
		titlefile = homedir() * "/GitHub/TopicModelsVB.jl/datasets/nsf/nsftitles.txt"
		corp = readcorp(docfile=docfile, vocabfile=vocabfile, titlefile=titlefile, counts=true)

	elseif corp_symbol == :citeu
		docfile = homedir() * "/GitHub/TopicModelsVB.jl/datasets/citeu/citeudocs.txt"
		vocabfile = homedir() * "/GitHub/TopicModelsVB.jl/datasets/citeu/citeuvocab.txt"
		userfile = homedir() * "/GitHub/TopicModelsVB.jl/datasets/citeu/citeuusers.txt"
		titlefile = homedir() * "/GitHub/TopicModelsVB.jl/datasets/citeu/citeutitles.txt"
		corp = readcorp(docfile=docfile, vocabfile=vocabfile, userfile=userfile, titlefile=titlefile, counts=true, readers=true)

	#if corp_symbol == :nsf
	#	docfile = homedir() * "/.julia/$version/topicmodelsvb/datasets/nsf/nsfdocs.txt"
	#	vocabfile = homedir() * "/.julia/$version/topicmodelsvb/datasets/nsf/nsfvocab.txt"
	#	titlefile = homedir() * "/.julia/$version/topicmodelsvb/datasets/nsf/nsftitles.txt"
	#	corp = readcorp(docfile=docfile, vocabfile=vocabfile, titlefile=titlefile, counts=true)

	#elseif corp_symbol == :citeu
	#	docfile = homedir() * "/.julia/$version/topicmodelsvb/datasets/citeu/citeudocs.txt"
	#	vocabfile = homedir() * "/.julia/$version/topicmodelsvb/datasets/citeu/citeuvocab.txt"
	#	userfile = homedir() * "/.julia/$version/topicmodelsvb/datasets/citeu/citeuusers.txt"
	#	titlefile = homedir() * "/.julia/$version/topicmodelsvb/datasets/citeu/citeutitles.txt"
	#	corp = readcorp(docfile=docfile, vocabfile=vocabfile, userfile=userfile, titlefile=titlefile, counts=true, readers=true)
		
		# why was this necessary?
		#padcorp!(corp)

	else
		println("Included corpora:\n:nsf\n:citeu")
		corp = nothing
	end

	return corp
end

function writecorp(corp::Corpus; docfile::String="", vocabfile::String="", userfile::String="", titlefile::String="", delim::Char=',', counts::Bool=false, readers::Bool=false, ratings::Bool=false)	
	"Write a corpus."

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

function abridge_corp!(corp::Corpus, n::Integer=0)
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
		users = sort(collect(values(corp.users)))

		ukey_map = Dict(ukey_old => ukey_new for (ukey_old, ukey_new) in zip(ukeys, ukeys[sortperm(sortperm([corp.users[ukey] for ukey in ukeys]))]))
		corp.users = Dict(ukey => user for (ukey, user) in zip(ukeys, users))

		for doc in unique(corp)
			doc.readers = [ukey_map[r] for r in doc.readers]
		end
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
			doc.readers = [ukey_map[ukey] for ukey in doc.readers]
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

function remove_empty_docs!(corp::Corpus)
	"Documents with no terms are removed from the corpus."

	keep = Bool[length(doc.terms) > 0 for doc in corp]
	corp.docs = corp[keep]
	nothing
end

function remove_redundant!(corp::Corpus; vocab::Bool=true, users::Bool=true)
	"Remove vocab and/or user keys which map to redundant values."
	"Reassign Document term and/or reader keys."

	if vocab
		vkey_map = Dict()
		vocab_dict_inverse = Dict()
		vkeys = sort(collect(keys(corp.vocab)))
		for vkey in vkeys
			if corp.vocab[vkey] in keys(vocab_dict_inverse)
				vkey_map[vkey] = vocab_dict_inverse[corp.vocab[vkey]]
				delete!(corp.vocab, vkey)

			else
				vkey_map[vkey] = vkey
				vocab_dict_inverse[corp.vocab[vkey]] = vkey
			end
		end

		for doc in unique(corp)
			doc.terms = [vkey_map[vkey] for vkey in doc.terms]
		end
	end

	if users
		ukey_map = Dict()
		users_dict_inverse = Dict()
		ukeys = sort(collect(keys(corp.users)))
		for ukey in ukeys
			if corp.users[ukey] in keys(users_dict_inverse)
				ukey_map[ukey] = users_dict_inverse[corp.users[ukey]]
				delete!(corp.users, ukey)

			else
				ukey_map[ukey] = ukey
				users_dict_inverse[corp.users[ukey]] = ukey
			end
		end

		for doc in unique(corp)
			doc.readers = [ukey_map[ukey] for ukey in doc.readers]
		end
	end
	nothing
end

function stop_corp!(corp::Corpus)
	"Filter stop words in the associated corpus."

	version = "v$(VERSION.major).$(VERSION.minor)"	

	stop_words = vec(readdlm(pwd() * "/GitHub/TopicModelsVB.jl/datasets/stopwords.txt", String))
	#stop_words = vec(readdlm(pwd() * "/.julia/$version/topicmodelsvb/datasets/stopwords.txt", String))
	stop_keys = filter(vkey -> lowercase(corp.vocab[vkey]) in stop_words, collect(keys(corp.vocab)))
	
	for doc in unique(corp)
		keep = Bool[!(j in stop_keys) for j in doc.terms]
		doc.terms = doc.terms[keep]
		doc.counts = doc.counts[keep]
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
		doc_ukeys = Set(vcat([doc.readers for doc in corp]...))
		corp.users = Dict(ukey => corp.users[ukey] for ukey in intersect(keys(corp.users), doc_ukeys))
	end
	nothing
end

function trim_docs!(corp::Corpus; terms::Bool=true, readers::Bool=true)
	"Those vocab and/or user keys which appear in documents but not in the corpus dictionaries are removed from the documents."

	if terms
		doc_vkeys = Set(vcat([doc.terms for doc in corp]...))
		bogus_vkeys = setdiff(doc_vkeys, keys(corp.vocab))
		for doc in unique(corp)
			keep = Bool[!(j in bogus_vkeys) for j in doc.terms]
			doc.terms = doc.terms[keep]
			doc.counts = doc.counts[keep]
		end
	end

	if terms
		doc_ukeys = Set(vcat([doc.readers for doc in corp]...))
		bogus_ukeys = setdiff(doc_ukeys, keys(corp.users))
		for doc in unique(corp)
			keep = Bool[!(u in bogus_ukeys) for u in doc.readers]
			doc.readers = doc.readers[keep]
			doc.ratings = doc.ratings[keep]
		end
	end
	nothing
end

function fixcorp!(corp::Corpus; vocab::Bool=true, users::Bool=true, abridge_corp::Integer=0, alphabetize_corp::Bool=false, compact_corp::Bool=false, condense_corp::Bool=false, pad_corp::Bool=false, remove_empty_docs::Bool=false, remove_redundant::Bool=false, stop_corp::Bool=false, trim_corp::Bool=false)
	"Generic function to ensure that a Corpus object can be loaded ino a TopicModel object."
	"Contains optional keyword arguments."

	pad_corp ? pad_corp!(corp) : trim_docs!(corp)

	remove_empty_docs 	&& remove_empty_docs!(corp)
	condense_corp 		&& condense_corp!(corp)
	remove_redundant	&& remove_redundant!(corp)
	abridge_corp > 0 	&& abridge_corp!(corp)
	pad_corp 			&& pad_corp!(corp, vocab=vocab, users=users)
	trim_corp 			&& trim_corp!(corp, vocab=vocab, users=users)

	stop_corp 			&& stop_corp!(corp)
	alphabetize_corp 	&& alphabetize_corp!(corp, vocab=vocab, users=users)

	compact_corp!(corp)
	nothing
end
