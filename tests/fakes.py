"""In-process Mongo fakes for research queue tests.

Lifted from tests/test_portfolio_trading.py so this repo does not import
the portfolio planning package.
"""

from __future__ import annotations

class _Result:
    def __init__(self, upserted_id=None, modified_count=0):
        self.upserted_id = upserted_id
        self.modified_count = modified_count


class _Cursor:
    def __init__(self, docs):
        self.docs = list(docs)

    def sort(self, key, direction=None):
        if isinstance(key, list) and direction is None:
            for sort_key, sort_direction in reversed(key):
                self.docs = sorted(
                    self.docs,
                    key=lambda doc: doc.get(sort_key) or "",
                    reverse=sort_direction < 0,
                )
            return self
        self.docs = sorted(self.docs, key=lambda doc: doc.get(key) or "", reverse=direction < 0)
        return self

    def limit(self, n):
        self.docs = self.docs[:n]
        return self

    def __iter__(self):
        return iter(self.docs)

class _Collection:
    def __init__(self):
        self.docs = []

    def create_index(self, *args, **kwargs):
        return True

    def find_one(self, query=None, projection=None, sort=None):
        docs = list(self.find(query, projection))
        if sort:
            for key, direction in reversed(sort):
                docs = sorted(docs, key=lambda doc: doc.get(key) or "", reverse=direction < 0)
        return docs[0] if docs else None

    def find(self, query=None, projection=None):
        rows = [self._project(doc, projection) for doc in self.docs if self._matches(doc, query or {})]
        return _Cursor(rows)

    def distinct(self, field, query=None):
        values = []
        seen = set()
        for doc in self.docs:
            if not self._matches(doc, query or {}):
                continue
            value = doc.get(field)
            if value not in seen:
                seen.add(value)
                values.append(value)
        return values

    def update_one(self, query, update, upsert=False):
        for doc in self.docs:
            if self._matches(doc, query):
                doc.update(update.get("$set", {}))
                for key, value in update.get("$inc", {}).items():
                    doc[key] = doc.get(key, 0) + value
                return _Result(modified_count=1)
        if upsert:
            new_doc = {**query, **update.get("$setOnInsert", {}), **update.get("$set", {})}
            for key, value in update.get("$inc", {}).items():
                new_doc[key] = value
            self.docs.append(new_doc)
            return _Result(upserted_id="new")
        return _Result()

    def update_many(self, query, update):
        count = 0
        for doc in self.docs:
            if self._matches(doc, query):
                doc.update(update.get("$set", {}))
                for key, value in update.get("$inc", {}).items():
                    doc[key] = doc.get(key, 0) + value
                count += 1
        return _Result(modified_count=count)

    def find_one_and_update(self, query, update, sort=None, return_document=None):
        docs = list(self.find(query))
        if sort:
            for key, direction in reversed(sort):
                docs = sorted(docs, key=lambda doc: doc.get(key) or "", reverse=direction < 0)
        if not docs:
            return None
        target = docs[0]
        for doc in self.docs:
            if doc is target or self._matches(doc, {"task_id": target.get("task_id")}):
                doc.update(update.get("$set", {}))
                for key, value in update.get("$inc", {}).items():
                    doc[key] = doc.get(key, 0) + value
                return dict(doc)
        return None

    def replace_one(self, query, new_doc, upsert=False):
        for idx, doc in enumerate(self.docs):
            if self._matches(doc, query):
                self.docs[idx] = dict(new_doc)
                return _Result(modified_count=1)
        if upsert:
            self.docs.append(dict(new_doc))
            return _Result(upserted_id="new")
        return _Result()

    def insert_one(self, doc):
        self.docs.append(dict(doc))
        return _Result(upserted_id="new")

    def insert_many(self, docs):
        self.docs.extend(dict(doc) for doc in docs)
        return _Result(upserted_id="new")

    def delete_many(self, query):
        before = len(self.docs)
        self.docs = [doc for doc in self.docs if not self._matches(doc, query)]
        return _Result(modified_count=before - len(self.docs))

    def delete_one(self, query):
        for idx, doc in enumerate(self.docs):
            if self._matches(doc, query):
                del self.docs[idx]
                return _Result(modified_count=1)
        return _Result(modified_count=0)

    def count_documents(self, query=None):
        return sum(1 for doc in self.docs if self._matches(doc, query or {}))

    def _project(self, doc, projection):
        if not projection:
            return dict(doc)
        include = {key for key, value in projection.items() if value}
        if not include:
            return dict(doc)
        return {key: doc.get(key) for key in include if key in doc}

    def _matches(self, doc, query):
        if "$or" in query:
            rest = {key: value for key, value in query.items() if key != "$or"}
            if rest and not self._matches(doc, rest):
                return False
            return any(self._matches(doc, clause) for clause in query["$or"])
        for key, expected in query.items():
            value = doc.get(key)
            if isinstance(expected, dict):
                if "$exists" in expected:
                    exists = key in doc
                    if bool(expected["$exists"]) != exists:
                        return False
                if "$lte" in expected and not (value <= expected["$lte"]):
                    return False
                if "$lt" in expected and not (value < expected["$lt"]):
                    return False
                if "$gte" in expected and not (value >= expected["$gte"]):
                    return False
                if "$gt" in expected and not (value > expected["$gt"]):
                    return False
                if "$ne" in expected and value == expected["$ne"]:
                    return False
                if "$nin" in expected and value in expected["$nin"]:
                    return False
                if "$in" in expected:
                    if isinstance(value, list):
                        if not set(value).intersection(expected["$in"]):
                            return False
                    elif value not in expected["$in"]:
                        return False
                    continue
            elif isinstance(value, list):
                if expected not in value:
                    return False
            elif value != expected:
                return False
        return True


class _Db(dict):
    def __getitem__(self, name):
        if name not in self:
            self[name] = _Collection()
        return dict.__getitem__(self, name)

