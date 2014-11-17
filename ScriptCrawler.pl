#!/usr/bin/perl

# NOTE: This script may contain bugs, since many links cannot be successfully followed.

use strict;
use warnings;

use LWP::Simple;

$| = 1; # Equivalent to an automatic fflush(stdout) in C/C++

my $MIN_SCRIPT_LENGTH = 128;

open(my $html, '<', 'all_scripts.html') or die "Could not open file 'all_scripts.html': $!\n";

my $code = '';

while (my $line = <$html>)
{
    $code .= $line;
}

close($html);

while ($code =~ /<a href=\"\/Movie Scripts\/(.+?)\s*Script.html/g)
{
    my $name = $1;
    
    print "Crawling film '$name'... ";
    
    my $friendly_name = $name;
    $friendly_name =~ s/ /-/g;

    my $html = get("http://www.imsdb.com/scripts/$friendly_name.html");
    
    if ($html)
    {
        $html =~ /<pre><html><head>.*?<\/head><body>(.+?)<\/pre><\/table><br>/s;
        
        my $script = $1;
        
        if ($script and length $script > $MIN_SCRIPT_LENGTH)
        {
            open(my $out, '>', "scripts/$name.html");
            
            print $out $script;
            
            close($out);
            
            print "OK.\n";
        }
        else
        {
            print "no script found.\n";
        }
    }
    else
    {
        print "page not found.\n";
    }
}