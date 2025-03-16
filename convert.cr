TABLE = {} of Char => Char
File.each_line("config/CharTable.txt") do |line|
  a, b = line.split('=', 2)
  next if a.size < 1
  TABLE[a.chars.first] = b.chars.first
end

input = File.read("sample.txt").split("\n").map(&.strip).reject!(&.empty?).join("\n")
output = input.chars.join { |x| TABLE.fetch(x, x) }
File.write("sample.out", output)
