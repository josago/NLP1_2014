#!/usr/bin/perl

use strict;
use warnings;

use LWP::Simple;

my $n = 1;
my $N = 25282;

open(my $movies, '<', 'movies.txt') or die "Could not read file 'movies.txt': $!\n";
open(my $output, '>', 'database.csv') or die "Could not write into file 'database.csv': $!\n";

foreach my $number (<$movies>)
{
    my $code = get("http://www.imdb.com/title/tt$number");
    
    $code =~ /<span class=\"itemprop\" itemprop=\"name\">(.+?)<\/span>/s;

    my $TITLE = $1;
    
    $code =~ /<a href=\"\/year\/(\d{4})\/\?ref_=tt_ov_inf\"/s;
    
    my $YEAR = $1;
    
    $code =~ /<span itemprop=\"ratingValue\">([\d\.]+?)<\/span>/s;
    
    my $SCORE = $1;
    
    my $RATING = '-';
    
    if ($code =~ /<span itemprop=\"contentRating\">(.+?)<\/span>/s)
    {
        $RATING = $1;
    }
    
    $code =~ /itemprop=\"director\".+?itemprop=\"name\">(.+?)<\/span>/s;
    
    my $DIRECTOR = $1;
    
    $code =~ /<div class="inline canwrap" itemprop="description">.*?<p>(.+?)<em class="nobr">/s;
    
    my $SUMMARY = $1;
    
    $code =~ /<h4 class=\"inline\">Stars\:<\/h4>.*?itemprop=\"name\">(.+?)<\/span><\/a>(.*?itemprop=\"name\">(.+?)<\/span><\/a>)?(.*?itemprop=\"name\">(.+?)<\/span><\/a>)?.*?<span class=\"ghost\">/s;
    
    my $STARS;
    
    unless ($5)
    {
        unless ($3)
        {
            $STARS = $1;
        }
        else
        {
            $STARS = "$1,$3";
        }
    }
    else
    {
        $STARS = "$1,$3,$5";
    }
    
    print $output "$TITLE\t$YEAR\t$DIRECTOR\t$SCORE\t$STARS\t$RATING\t$SUMMARY\n";
    
    print "Parsing movie $n out of $N...\n";
    $n++;
}

close $movies;
close $output;