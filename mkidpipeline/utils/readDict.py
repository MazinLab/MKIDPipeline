def ask_for(key):
    s = input("ReadDict: enter value for '%s': " % key)
    try:
        val = eval(s)
    except NameError:
        # allow people to enter unquoted strings
        val = s
    return val


class readDict(dict):

    def __init__(self, ask=False):
        """
        @param ask if the dict doesn't have an entry for a key, ask for the associated value and assign
        """
        dict.__init__(self)
        self.ask = ask

    def __getitem__(self, key):
        if key not in self:
            if self.ask:
                print("ReadDict: parameter '%s' not found" % key)
                val = ask_for(key)
                print("ReadDict: setting '%s' = %s" % (key, repr(val)))
                dict.__setitem__(self, key, val)
            else:
                return None
        return dict.__getitem__(self, key)

    def read_from_file(self, filename):
        f = open(filename)
        old = ''
        for line in f:
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue
            s = line.split('#')
            line = s[0]
            s = line.split('\\')
            if len(s) > 1:
                # old = string.join([old, s[0]]) string.join depricated in p3 - replace
                old.join(s[0])
                continue
            else:
                # line = string.join([old, s[0]])
                old = ''
            for i in range(len(line)):
                if line[i] != ' ':
                    line = line[i:]
                    break
            exec (line)
            s = line.split('=')
            if len(s) != 2:
                print("Error parsing line:")
                print(line)
                continue
            key = s[0].strip()
            val = eval(s[1].strip())  # XXX:make safer
            self[key] = val
        f.close()

    readFromFile = read_from_file
