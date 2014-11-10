#!/usr/bin/perl

use strict;
use warnings;

use LWP::Simple;

my %movies;

open(my $file, '>', 'movies.txt') or die "Could not write in file 'movies.txt': $!\n";

for my $page (1 .. 100)
{
    print "Currently working on page $page...\n";

    for my $search ('a' .. 'z')
    {
        my $start = 100 * ($page - 1) + 1;
        
        my $code  = get("http://www.imdb.com/search/title?count=100&start=$start&title=$search&title_type=feature");
        
        while ($code =~ /href=\"\/title\/tt(.+)\/\"/g)
        {
            unless (exists $movies{$1})
            {
                $movies{$1} = 1;
                
                print $file "$1\n";
            }
        }
        
        print "\tCrawled movies so far: " . keys(%movies) . "\n";
    }
}

close $file;