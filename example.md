### Key Tables for Testing

1. **SOOrder**: contains `OrderNbr`, `CustomerID`, `OrderDate`, `OrderTotal`, `Status`
2. **SOShipment**: contains `ShipmentNbr`, `ShipmentDate`, `CustomerID`, links to `OrderNbr`
3. **InTran**: inventory movements — includes `InventoryID`, `Qty`, `TranDate`, `LocationID`
4. **InventoryItem**: product details — `InventoryID`, `Descr`, `ItemClassID`, `BasePrice`

---

### 5 Example CURL Requests

#### 1. Tổng doanh số bán hàng trong tháng 6 năm 2025

```bash
curl -X POST https://binkhoale1812-cpg-chatbot.hf.space/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Tổng doanh số bán hàng trong tháng 6 năm 2025 là bao nhiêu?"}'
```

#### 2. Có bao nhiêu đơn hàng được tạo vào ngày 2 tháng 6 năm 2025

```bash
curl -X POST https://binkhoale1812-cpg-chatbot.hf.space/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Có bao nhiêu đơn hàng được tạo vào ngày 2 tháng 6 năm 2025?"}'
```

#### 3. Những sản phẩm nào được bán nhiều nhất trong tháng 6

```bash
curl -X POST https://binkhoale1812-cpg-chatbot.hf.space/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Những sản phẩm nào được bán nhiều nhất trong tháng 6?"}'
```

#### 4. Bao nhiêu sản phẩm ID 4836 đã được xuất kho

```bash
curl -X POST https://binkhoale1812-cpg-chatbot.hf.space/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Bao nhiêu sản phẩm có mã 4836 đã được xuất kho?"}'
```

#### 5. Giá trung bình của từng nhóm sản phẩm là bao nhiêu?

```bash
curl -X POST https://binkhoale1812-cpg-chatbot.hf.space/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Giá trung bình của từng nhóm sản phẩm là bao nhiêu?"}'
```