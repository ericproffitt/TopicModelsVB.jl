library(ggplot2)

# 12 x 9
df = data.frame(x=c(1,2,3.5,4.5,6,7), label=c('gpuLDA', 'LDA', 'gpuCTM', 'CTM', 'gpuCTPF', 'CTPF'), Seconds=c(32, 404, 115, 1974, 27, 831), Processor=rep(c('Intel Iris Plus Graphics 640', '2.3 GHz Dual-Core Intel Core i5'), 3))
p = ggplot(df, aes(x=x, y=Seconds, fill=Processor)) +
    geom_bar(stat='identity', width=0.975) +
    scale_y_log10() +
    scale_fill_manual(values=c('#5168ed', '#ff4a4a')) +
    scale_x_continuous(breaks=df$x, label=df$label) +
    #scale_y_continuous(breaks=c(0,500,1000,1500,2000)) +
    theme(axis.title.x=element_blank(), axis.line.x=element_line(color='grey'), axis.ticks.x=element_blank()) +
    theme(axis.title.y=element_text(color='#424242', size=10), axis.line.y=element_line(color='grey'), axis.text.y=element_text(color='#424242', size=8), axis.ticks.y=element_line(color='grey')) +
    theme(legend.title=element_text(color='#424242'), legend.text=element_text(color='#424242')) +
    theme(plot.background=element_rect(fill='white', color=NA), panel.background=element_rect(fill='white'), panel.grid.major=element_blank(), panel.grid.minor=element_blank()) +
    theme(plot.caption=element_text(color='#424242', hjust=1, vjust=-6, lineheight=0.5)) +
    theme(plot.margin=unit(c(0.5,0.5,1.05,0.5), 'cm')) +
    labs(caption="LDA and CTM run for 10 iterations : 100,000 NSF documents with 30 topics.\n\nCTPF run for 10 iterations : CiteULike dataset with 100 topics.")

p

df = data.frame(p=1-x)

p = ggplot(df, aes(x=p))+
    geom_histogram(color="white", fill="#5168ed", bins=50) +
    scale_x_continuous(breaks=0:10/10) +
    scale_y_continuous(breaks=c(0,1000,2000,3000,4000,5000)) +
    theme(axis.title.x=element_text(color='#424242', size=10), axis.line.x=element_line(color='grey'), axis.text.x=element_text(color='#424242', size=8), axis.ticks.x=element_line(color='grey')) +
    theme(axis.title.y=element_text(color='#424242', size=10), axis.line.y=element_line(color='grey'), axis.text.y=element_text(color='#424242', size=8), axis.ticks.y=element_line(color='grey')) +
    theme(legend.title=element_text(color='#424242'), legend.text=element_text(color='#424242')) +
    xlab("\nCiteULike Science Article Database : Collaborative topic Poisson factorization model with 100 topics.") +
    ylab("Documents") +
    theme(plot.background=element_rect(fill='white', color=NA), panel.background=element_rect(fill='white'), panel.grid.major=element_blank(), panel.grid.minor=element_blank())

p


